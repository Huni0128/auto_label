"""Dataset conversion runner used by the GUI."""
from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from PyQt5.QtCore import QRunnable

from ..core.imaging import letterbox
from ..qt.signals import Signals

random.seed(0)


@dataclass(frozen=True)
class ConvertConfig:
    size: Union[int, Tuple[int, int]] = (1280, 720)
    val_ratio: float = 0.2


def _norm_xy_rect(x: float, y: float, out_w: int, out_h: int) -> Tuple[float, float]:
    return x / float(out_w), y / float(out_h)


def _rect_to_polygon(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    (x1, y1), (x2, y2) = points
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _poly_to_yolo_line(
    cls_id: int,
    polygon: Sequence[Tuple[float, float]],
    out_w: int,
    out_h: int,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> str:
    coords: List[str] = []
    for x, y in polygon:
        xx = x * scale + pad_w
        yy = y * scale + pad_h
        nx, ny = _norm_xy_rect(xx, yy, out_w, out_h)
        coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return f"{cls_id} " + " ".join(coords)


def _save_pair(split: str, out_root: Path, stem: str, image: Image.Image, lines: Sequence[str]) -> None:
    img_out = out_root / f"images/{split}/{stem}.jpg"
    lab_out = out_root / f"labels/{split}/{stem}.txt"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(img_out), quality=95)
    lab_out.write_text("\n".join(lines), encoding="utf-8")


def _detect_coco(json_dir: Path) -> Optional[Dict]:
    for path in json_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if all(key in obj for key in ("images", "annotations", "categories")):
            return obj
    return None


def _iter_labelme_files(json_dir: Path) -> Iterable[Path]:
    return json_dir.glob("*.json")


def _emit(signals: Signals | None, ok: bool, message: str) -> None:
    if signals:
        signals.one_done.emit(ok, message)


def _load_labelme_metadata(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class DatasetConvertRunner(QRunnable):
    """Background worker that builds a YOLO segmentation dataset."""

    def __init__(
        self,
        img_dir: Path,
        json_dir: Path,
        output_root: Path,
        config: ConvertConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.output_root = output_root
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    # Helpers -----------------------------------------------------------------
    def _should_stop(self) -> bool:
        return self.stop_event.is_set() if self.stop_event else False

    def _save_dataset_yaml(self, class_names: Sequence[str]) -> None:
        content = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            f"names: {list(class_names)}\n"
        )
        (self.output_root / "dataset.yaml").write_text(content, encoding="utf-8")

    # Dataset pipelines -------------------------------------------------------
    def _process_coco(self, dataset: Dict) -> None:
        categories = sorted(dataset["categories"], key=lambda cat: cat["id"])
        category_id_map = {category["id"]: idx for idx, category in enumerate(categories)}
        class_names = [category["name"] for category in categories]

        from collections import defaultdict

        annotations = defaultdict(list)
        for ann in dataset["annotations"]:
            annotations[ann["image_id"]].append(ann)

        for image_info in dataset["images"]:
            if self._should_stop():
                break

            file_name = Path(image_info.get("file_name", "")).name
            src = self.img_dir / file_name
            if not src.exists():
                alt = Path(image_info.get("file_name", ""))
                src = alt if alt.exists() else None
            if not src or not src.exists():
                _emit(self.signals, False, f"[MISS] {file_name}")
                continue

            try:
                image = Image.open(src).convert("RGB")
            except Exception as exc:
                _emit(self.signals, False, f"[FAIL] open {file_name} - {exc}")
                continue

            resized, scale, pad_w, pad_h, _ = letterbox(image, self.config.size)
            out_w, out_h = resized.size

            lines: List[str] = []
            for ann in annotations.get(image_info["id"], []):
                cls = category_id_map[ann["category_id"]]
                segmentation = ann.get("segmentation")
                polygons: List[List[Tuple[float, float]]] = []
                if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
                    for flat in segmentation:
                        pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
                        if len(pts) >= 3:
                            polygons.append(pts)
                else:
                    x, y, w, h = ann["bbox"]
                    polygons = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]

                for polygon in polygons:
                    if len(polygon) >= 3:
                        lines.append(
                            _poly_to_yolo_line(cls, polygon, out_w, out_h, scale, pad_w, pad_h)
                        )

            split = "val" if random.random() < self.config.val_ratio else "train"
            _save_pair(split, self.output_root, Path(file_name).stem, resized, lines)
            _emit(self.signals, True, f"[OK] {file_name}")

        self._save_dataset_yaml(class_names)

    def _process_labelme(self) -> None:
        metadata = []
        class_names = set()
        for json_path in _iter_labelme_files(self.json_dir):
            meta = _load_labelme_metadata(json_path)
            if not meta:
                continue
            for shape in meta.get("shapes", []):
                class_names.add(shape.get("label"))
            metadata.append((json_path, meta))

        names_sorted = sorted(n for n in class_names if n)
        class_index = {name: idx for idx, name in enumerate(names_sorted)}

        for json_path, meta in metadata:
            if self._should_stop():
                break

            image_name = meta.get("imagePath") or f"{json_path.stem}.jpg"
            stem = Path(image_name).stem

            src = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"]:
                candidate = self.img_dir / f"{stem}{ext}"
                if candidate.exists():
                    src = candidate
                    break
            if not src:
                _emit(self.signals, False, f"[MISS] {stem}.*")
                continue

            try:
                image = Image.open(src).convert("RGB")
            except Exception as exc:
                _emit(self.signals, False, f"[FAIL] open {src.name} - {exc}")
                continue

            resized, scale, pad_w, pad_h, _ = letterbox(image, self.config.size)
            out_w, out_h = resized.size

            lines: List[str] = []
            for shape in meta.get("shapes", []):
                label = shape.get("label")
                if label not in class_index:
                    continue
                points = shape.get("points", [])
                if shape.get("shape_type", "polygon") == "rectangle" and len(points) == 2:
                    polygon = _rect_to_polygon(points)
                else:
                    polygon = [(p[0], p[1]) for p in points]
                if len(polygon) < 3:
                    continue
                cls_id = class_index[label]
                lines.append(_poly_to_yolo_line(cls_id, polygon, out_w, out_h, scale, pad_w, pad_h))

            split = "val" if random.random() < self.config.val_ratio else "train"
            _save_pair(split, self.output_root, stem, resized, lines)
            _emit(self.signals, True, f"[OK] {stem}")

        self._save_dataset_yaml(names_sorted)

    # QRunnable ---------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
                (self.output_root / sub).mkdir(parents=True, exist_ok=True)

            coco = _detect_coco(self.json_dir)
            if coco:
                self._process_coco(coco)
            else:
                self._process_labelme()
        except Exception as exc:  # pragma: no cover - defensive
            _emit(self.signals, False, f"[FAIL] convert - {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()