"""LabelMe JSON augmentation worker."""
from __future__ import annotations

import base64
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PyQt5.QtCore import QRunnable

from ..core.constants import AUGMENT_RESOLUTION, IMG_EXTS
from ..core.polygons import flatten_polygons, sanitize_polygon
from ..qt.signals import Signals


@dataclass(frozen=True)
class AugmentationConfig:
    multiplier: int
    min_polygon_area: float = 5.0
    target_width: int = AUGMENT_RESOLUTION[0]
    target_height: int = AUGMENT_RESOLUTION[1]

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.target_width, self.target_height)


def build_transforms(config: AugmentationConfig) -> A.Compose:
    """Return an Albumentations pipeline configured for the GUI defaults."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.0),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.9,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Resize(config.target_height, config.target_width, interpolation=cv2.INTER_AREA, p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _decode_image_data(image_data: str) -> np.ndarray | None:
    try:
        data = base64.b64decode(image_data)
    except Exception:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _resolve_image_path(meta: dict, json_path: Path, input_dir: Path) -> Path | None:
    if isinstance(meta.get("imagePath"), str) and meta["imagePath"]:
        candidate = (input_dir / meta["imagePath"]).resolve()
        if candidate.exists():
            return candidate
    stem = json_path.stem
    for ext in IMG_EXTS:
        candidate = (input_dir / f"{stem}{ext}").resolve()
        if candidate.exists():
            return candidate
    return None


class LabelmeAugmentationTask(QRunnable):
    """Create augmented copies of a single LabelMe sample."""

    def __init__(
        self,
        json_path: Path,
        input_dir: Path,
        output_dir: Path,
        config: AugmentationConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    def _emit(self, ok: bool, message: str) -> None:
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
        if self.stop_event.is_set():
            self._emit(False, f"[STOPPED] {self.json_path.name}")
            return

        try:
            meta = json.loads(self.json_path.read_text(encoding="utf-8"))
            image_path = _resolve_image_path(meta, self.json_path, self.input_dir)

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
            if image is None and meta.get("imageData"):
                image = _decode_image_data(meta["imageData"])
            if image is None:
                self._emit(False, f"[FAIL] 이미지 로드 실패: {self.json_path.name}")
                return

            shapes = meta.get("shapes", [])
            polygons: list[list[tuple[float, float]]] = []
            polygon_indices: list[int] = []
            for idx, shape in enumerate(shapes):
                if shape.get("shape_type", "polygon") != "polygon":
                    continue
                pts = shape.get("points", [])
                if not pts or len(pts) < 3:
                    continue
                polygons.append([(float(x), float(y)) for x, y in pts])
                polygon_indices.append(idx)

            flat_keypoints, layout = flatten_polygons(polygons)
            base_stem = image_path.stem if image_path else self.json_path.stem
            output_ext = image_path.suffix if image_path else ".png"

            pipeline = build_transforms(self.config)

            for idx in range(max(1, self.config.multiplier)):
                if self.stop_event.is_set():
                    self._emit(False, f"[STOPPED] {self.json_path.stem} (k={idx + 1})")
                    break

                keypoints_in = flat_keypoints if flat_keypoints else [(0.0, 0.0)]
                transformed = pipeline(image=image, keypoints=keypoints_in)
                aug_image = transformed["image"]
                aug_keypoints = transformed["keypoints"]
                height, width = aug_image.shape[:2]

                new_shapes = deepcopy(shapes)
                if flat_keypoints and len(aug_keypoints) >= 1:
                    keypoints_np = np.array(aug_keypoints, dtype=np.float32)
                    for layout_idx, (start, end) in enumerate(layout):
                        pts_aug = [(float(keypoints_np[i, 0]), float(keypoints_np[i, 1])) for i in range(start, end)]
                        pts_aug = sanitize_polygon(pts_aug, width, height, self.config.min_polygon_area)
                        target_idx = polygon_indices[layout_idx]
                        if len(pts_aug) >= 3:
                            new_shapes[target_idx]["points"] = [[float(x), float(y)] for x, y in pts_aug]
                            new_shapes[target_idx]["shape_type"] = "polygon"
                        else:
                            new_shapes[target_idx]["points"] = []
                    new_shapes = [
                        s
                        for s in new_shapes
                        if not (s.get("shape_type", "polygon") == "polygon" and len(s.get("points", [])) < 3)
                    ]
                else:
                    new_shapes = [s for s in new_shapes if s.get("shape_type", "polygon") != "polygon"]

                output_image_name = f"{base_stem}_aug{idx + 1}{output_ext}"
                output_json_name = f"{base_stem}_aug{idx + 1}.json"

                self.output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.output_dir / output_image_name), aug_image)

                new_meta = deepcopy(meta)
                new_meta["imagePath"] = output_image_name
                new_meta["imageWidth"] = int(width)
                new_meta["imageHeight"] = int(height)
                new_meta["imageData"] = None
                new_meta["shapes"] = new_shapes

                (self.output_dir / output_json_name).write_text(
                    json.dumps(new_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                self._emit(True, f"[OK] {output_image_name}")
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(False, f"[FAIL] {self.json_path.name} - {exc}")