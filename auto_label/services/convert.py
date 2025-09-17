"""GUI에서 사용하는 데이터셋 변환 러너.

- COCO 또는 LabelMe 주석을 입력으로 받아 YOLO segmentation 형식으로 변환합니다.
- 이미지/라벨을 train/val 디렉터리로 분할·저장하고, dataset.yaml을 생성합니다.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
from PyQt5.QtCore import QRunnable

from ..core.imaging import letterbox
from ..qt.signals import Signals

# 재현성 확보를 위한 고정 시드
random.seed(0)


@dataclass(frozen=True)
class ConvertConfig:
    """데이터셋 변환 설정 값.

    Attributes:
        size: 리사이즈 타깃 (w, h) 또는 정수(정사각형).
        val_ratio: 검증 세트 비율(0~1).
    """
    size: int | tuple[int, int] = (1280, 720)
    val_ratio: float = 0.2


def _norm_xy_rect(x: float, y: float, out_w: int, out_h: int) -> tuple[float, float]:
    """출력 크기 기준 정규화 좌표를 반환합니다."""
    return x / float(out_w), y / float(out_h)


def _rect_to_polygon(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """(x1, y1), (x2, y2) 사각형을 시계 사각형 폴리곤으로 변환합니다."""
    (x1, y1), (x2, y2) = points
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _poly_to_yolo_line(
    cls_id: int,
    polygon: Sequence[tuple[float, float]],
    out_w: int,
    out_h: int,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> str:
    """폴리곤을 YOLO-SEG 한 줄 문자열로 변환합니다."""
    coords: list[str] = []
    for x, y in polygon:
        xx = x * scale + pad_w
        yy = y * scale + pad_h
        nx, ny = _norm_xy_rect(xx, yy, out_w, out_h)
        coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return f"{cls_id} " + " ".join(coords)


def _save_pair(
    split: str,
    out_root: Path,
    stem: str,
    image: Image.Image,
    lines: Sequence[str],
) -> None:
    """이미지/라벨 파일을 지정 split 경로에 저장합니다."""
    img_out = out_root / f"images/{split}/{stem}.jpg"
    lab_out = out_root / f"labels/{split}/{stem}.txt"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(img_out), quality=95)
    lab_out.write_text("\n".join(lines), encoding="utf-8")


def _detect_coco(json_dir: Path) -> dict[str, Any] | None:
    """COCO 포맷 JSON을 탐지하여 로드합니다(첫 번째 일치 항목)."""
    for path in json_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if all(key in obj for key in ("images", "annotations", "categories")):
            return obj  # type: ignore[return-value]
    return None


def _iter_labelme_files(json_dir: Path) -> Iterable[Path]:
    """LabelMe JSON 파일을 이터레이션합니다."""
    return json_dir.glob("*.json")


def _emit(signals: Signals | None, ok: bool, message: str) -> None:
    """UI로 진행/결과 메시지를 전송합니다."""
    if signals:
        signals.one_done.emit(ok, message)


def _load_labelme_metadata(path: Path) -> dict[str, Any] | None:
    """LabelMe 메타데이터를 안전하게 로드합니다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class DatasetConvertRunner(QRunnable):
    """YOLO segmentation 데이터셋을 구성하는 백그라운드 작업."""

    def __init__(
        self,
        img_dir: Path,
        json_dir: Path,
        output_root: Path,
        config: ConvertConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        """작업 인스턴스를 초기화합니다."""
        super().__init__()
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.output_root = output_root
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    # Helpers -----------------------------------------------------------------
    def _should_stop(self) -> bool:
        """중단 이벤트가 설정되었는지 확인합니다."""
        return self.stop_event.is_set() if self.stop_event else False

    def _save_dataset_yaml(self, class_names: Sequence[str]) -> None:
        """YOLO dataset.yaml을 생성합니다."""
        content = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            f"names: {list(class_names)}\n"
        )
        (self.output_root / "dataset.yaml").write_text(content, encoding="utf-8")

    # Dataset pipelines -------------------------------------------------------
    def _process_coco(self, dataset: dict[str, Any]) -> None:
        """COCO 포맷을 YOLO-SEG 형식으로 변환합니다."""
        categories = sorted(dataset["categories"], key=lambda cat: cat["id"])
        category_id_map = {
            category["id"]: idx for idx, category in enumerate(categories)
        }
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

            lines: list[str] = []
            for ann in annotations.get(image_info["id"], []):
                cls = category_id_map[ann["category_id"]]
                segmentation = ann.get("segmentation")
                polygons: list[list[tuple[float, float]] | None] = []

                if (
                    isinstance(segmentation, list)
                    and segmentation
                    and isinstance(segmentation[0], list)
                ):
                    for flat in segmentation:
                        pts = [
                            (flat[i], flat[i + 1])
                            for i in range(0, len(flat), 2)
                        ]
                        polygons.append(pts if len(pts) >= 3 else None)
                else:
                    x, y, w, h = ann["bbox"]
                    rect = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    polygons.append(rect)

                for polygon in polygons:
                    if polygon and len(polygon) >= 3:
                        lines.append(
                            _poly_to_yolo_line(
                                cls, polygon, out_w, out_h, scale, pad_w, pad_h
                            )
                        )

            split = "val" if random.random() < self.config.val_ratio else "train"
            _save_pair(
                split,
                self.output_root,
                Path(file_name).stem,
                resized,
                lines,
            )
            _emit(self.signals, True, f"[OK] {file_name}")

        self._save_dataset_yaml(class_names)

    def _process_labelme(self) -> None:
        """LabelMe 포맷을 YOLO-SEG 형식으로 변환합니다."""
        metadata: list[tuple[Path, dict[str, Any]]] = []
        class_names: set[str | None] = set()

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

            src: Path | None = None
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

            lines: list[str] = []
            for shape in meta.get("shapes", []):
                label = shape.get("label")
                if label not in class_index:
                    continue

                points = shape.get("points", [])
                if (
                    shape.get("shape_type", "polygon") == "rectangle"
                    and len(points) == 2
                ):
                    polygon = _rect_to_polygon(points)
                else:
                    polygon = [(float(p[0]), float(p[1])) for p in points]

                if len(polygon) < 3:
                    continue

                cls_id = class_index[label]
                lines.append(
                    _poly_to_yolo_line(
                        cls_id,
                        polygon,
                        out_w,
                        out_h,
                        scale,
                        pad_w,
                        pad_h,
                    )
                )

            split = "val" if random.random() < self.config.val_ratio else "train"
            _save_pair(split, self.output_root, stem, resized, lines)
            _emit(self.signals, True, f"[OK] {stem}")

        self._save_dataset_yaml(names_sorted)

    # QRunnable ---------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - Qt thread pool에서 실행
        """작업 실행 진입점.

        - 출력 디렉터리 초기화
        - COCO 감지 후 해당 파이프라인 실행, 없으면 LabelMe 파이프라인 실행
        """
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
                (self.output_root / sub).mkdir(parents=True, exist_ok=True)

            coco = _detect_coco(self.json_dir)
            if coco:
                self._process_coco(coco)
            else:
                self._process_labelme()

        except Exception as exc:  # pragma: no cover - 방어적 처리
            _emit(self.signals, False, f"[FAIL] convert - {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()
