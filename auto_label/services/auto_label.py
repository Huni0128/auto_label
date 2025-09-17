"""Automatic labelling runner using Ultralytics YOLO segmentation models."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QRunnable

from ..core.imaging import clip_points
from ..core.files import copy_file, list_image_files
from ..core.yolo import (
    draw_viz,
    get_ultra_segments,
    save_yolo_seg_txt,
    score_to_bucket,
    to_labelme_shapes,
)
from ..qt.signals import Signals

try:  # pragma: no cover - optional dependency at runtime
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


@dataclass(frozen=True)
class AutoLabelConfig:
    model_path: Path
    image_root: Path
    save_root: Path
    conf: float
    iou: float
    imgsz_w: int
    imgsz_h: int
    device: str | None
    approx_eps: float
    min_area: float
    viz: bool
    copy_images: bool
    copy_mode: str

    @property
    def imgsz(self) -> int | tuple[int, int]:
        return (self.imgsz_h, self.imgsz_w) if self.imgsz_w != self.imgsz_h else int(self.imgsz_w)


class AutoLabelRunner(QRunnable):
    """Background runnable that performs segmentation inference and exports labels."""

    def __init__(self, config: AutoLabelConfig, stop_event: threading.Event, signals: Signals) -> None:
        super().__init__()
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    # ------------------------------------------------------------------ utils
    def _emit(self, ok: bool, message: str) -> None:
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def _ensure_model(self):
        if YOLO is None:
            raise RuntimeError("`ultralytics` 패키지가 설치되어 있지 않습니다") from _YOLO_IMPORT_ERROR
        return YOLO(str(self.config.model_path))

    def _load_class_names(self, model) -> List[str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return [names[idx] for idx in sorted(names.keys())]
        if isinstance(names, list):
            return names
        return []

    # ------------------------------------------------------------------ runner
    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
        try:
            images = list_image_files(self.config.image_root)
        except Exception as exc:
            self._emit(False, f"[ERR] 입력 이미지 검색 실패: {exc}")
            if self.signals:
                self.signals.all_done.emit()
            return

        if not images:
            self._emit(False, "[INFO] 처리할 이미지가 없습니다.")
            if self.signals:
                self.signals.all_done.emit()
            return

        try:
            model = self._ensure_model()
            class_names = self._load_class_names(model)
        except Exception as exc:
            self._emit(False, f"[ERR] 모델 로드 실패: {exc}")
            if self.signals:
                self.signals.all_done.emit()
            return

        total = len(images)
        for index, image_path in enumerate(images, 1):
            if self.stop_event.is_set():
                self._emit(False, "[STOP] 사용자 중지 요청")
                break

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                self._emit(False, f"[{image_path.name}] 이미지 열기 실패: {exc}")
                continue

            try:
                predictions = model.predict(
                    source=str(image_path),
                    imgsz=self.config.imgsz,
                    conf=float(self.config.conf),
                    iou=float(self.config.iou),
                    device=self.config.device or None,
                    verbose=False,
                )
            except Exception as exc:
                self._emit(False, f"[{image_path.name}] 예측 실패: {exc}")
                continue

            if not predictions:
                self._handle_no_prediction(image_path, index, total)
                continue

            result = predictions[0]
            shapes: List[dict] = []
            if getattr(result, "masks", None) is not None and result.masks and result.masks.data is not None:
                mask_count = int(result.masks.data.shape[0])
                cls_ids = (
                    result.boxes.cls.cpu().numpy().astype(int)
                    if result.boxes is not None and result.boxes.cls is not None
                    else np.zeros((mask_count,), dtype=int)
                )
                confs = (
                    result.boxes.conf.cpu().numpy()
                    if result.boxes is not None and result.boxes.conf is not None
                    else np.ones((mask_count,), dtype=float)
                )

                for idx in range(mask_count):
                    class_id = int(cls_ids[idx]) if idx < len(cls_ids) else 0
                    score = float(confs[idx]) if idx < len(confs) else 1.0
                    label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                    polygons = get_ultra_segments(result, idx, approx_eps=float(self.config.approx_eps))
                    if polygons:
                        clipped = [clip_points(poly, width, height) for poly in polygons]
                        shapes.extend(
                            to_labelme_shapes(clipped, label, class_id, score, float(self.config.min_area))
                        )

            if shapes:
                self._save_with_shapes(image_path, width, height, shapes, index, total)
            else:
                self._handle_no_prediction(image_path, index, total)

        if self.signals:
            self.signals.all_done.emit()

    # ----------------------------------------------------------------- helpers
    def _handle_no_prediction(
        self,
        image_path: Path,
        index: int | None = None,
        total: int | None = None,
    ) -> None:
        bucket = "lt_0.60"
        txt_dir = self.config.save_root / bucket / "yolo-seg"
        viz_dir = self.config.save_root / bucket / "viz"
        txt_path = txt_dir / f"{image_path.stem}.txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("", encoding="utf-8")

        if self.config.copy_images:
            try:
                copy_file(image_path, txt_dir / image_path.name, self.config.copy_mode)
            except Exception as exc:
                self._emit(False, f"[{image_path.name}] 이미지 보관 실패: {exc}")

        if self.config.viz:
            img = cv2.imread(str(image_path))
            if img is not None:
                viz_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(viz_dir / image_path.name), img)

        progress = f" ({index}/{total})" if index is not None and total is not None else ""
        self._emit(True, f"[{image_path.name}] segments=0 -> bin={bucket} -> SAVE empty .txt{progress}")

    def _save_with_shapes(
        self,
        image_path: Path,
        width: int,
        height: int,
        shapes: Sequence[dict],
        index: int,
        total: int,
    ) -> None:
        scores = [float(shape.get("flags", {}).get("score", 1.0)) for shape in shapes]
        aggregate = min(scores) if scores else 1.0
        bucket = score_to_bucket(aggregate)

        txt_dir = self.config.save_root / bucket / "yolo-seg"
        viz_dir = self.config.save_root / bucket / "viz"
        txt_path = txt_dir / f"{image_path.stem}.txt"

        save_yolo_seg_txt(txt_path, width, height, shapes)

        if self.config.copy_images:
            try:
                copy_file(image_path, txt_dir / image_path.name, self.config.copy_mode)
            except Exception as exc:
                self._emit(False, f"[{image_path.name}] 이미지 보관 실패: {exc}")

        if self.config.viz:
            image = cv2.imread(str(image_path))
            if image is not None:
                visualised = draw_viz(image, shapes)
                viz_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(viz_dir / image_path.name), visualised)

        self._emit(
            True,
            f"[{image_path.name}] segments={len(shapes)} score(min)={aggregate:.3f} -> bin={bucket}"
            f" ({index}/{total})",
        )