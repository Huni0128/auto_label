"""Ultralytics YOLO 세그멘테이션 모델을 이용한 자동 라벨링 러너.

- 지정 폴더 내 이미지를 순회하며 YOLO segmentation 추론을 수행합니다.
- 결과 폴리곤을 LabelMe 호환 형태로 변환한 뒤 YOLO-SEG 텍스트 형식으로 저장합니다.
- 집계 점수에 따라 결과를 버킷 디렉터리로 분류하고, 옵션에 따라 원본/시각화 이미지를 보관합니다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QRunnable

from ..core.files import copy_file, list_image_files
from ..core.imaging import clip_points
from ..core.yolo import (
    draw_viz,
    get_ultra_segments,
    save_yolo_seg_txt,
    score_to_bucket,
    to_labelme_shapes,
)
from ..qt.signals import Signals

try:  # pragma: no cover - 선택적 런타임 의존성
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - 런타임 처리
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


@dataclass(frozen=True)
class AutoLabelConfig:
    """오토 라벨링 실행을 위한 설정 값.

    Attributes:
        model_path: YOLO 가중치(.pt) 경로.
        image_root: 입력 이미지 루트 디렉터리.
        save_root: 결과 저장 루트 디렉터리.
        conf: confidence threshold.
        iou: IoU threshold.
        imgsz_w: 입력 리사이즈 가로.
        imgsz_h: 입력 리사이즈 세로.
        device: 추론 디바이스(e.g., "cpu", "0").
        approx_eps: 폴리곤 근사화 epsilon.
        min_area: 최소 폴리곤 면적(필터링).
        viz: 시각화 이미지 저장 여부.
        copy_images: 결과 버킷에 원본 이미지 보관 여부.
        copy_mode: 보관 시 파일 처리 모드("copy", "link" 등).
    """

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
        """Ultralytics `imgsz` 인자 형식으로 반환합니다."""
        return (
            (self.imgsz_h, self.imgsz_w)
            if self.imgsz_w != self.imgsz_h
            else int(self.imgsz_w)
        )


class AutoLabelRunner(QRunnable):
    """세그멘테이션 추론을 수행하고 라벨을 저장하는 백그라운드 작업."""

    def __init__(
        self,
        config: AutoLabelConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        """작업 인스턴스를 초기화합니다."""
        super().__init__()
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    # ------------------------------------------------------------------ utils
    def _emit(self, ok: bool, message: str) -> None:
        """UI로 진행/결과 메시지를 전송합니다."""
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def _ensure_model(self):
        """Ultralytics YOLO 모델을 로드합니다."""
        if YOLO is None:
            raise RuntimeError("`ultralytics` 패키지가 필요합니다.") from _YOLO_IMPORT_ERROR
        return YOLO(str(self.config.model_path))

    def _load_class_names(self, model) -> List[str]:
        """모델의 클래스 이름 목록을 반환합니다."""
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return [names[idx] for idx in sorted(names.keys())]
        if isinstance(names, list):
            return names
        return []

    # ------------------------------------------------------------------ runner
    def run(self) -> None:  # pragma: no cover - Qt thread pool에서 실행
        """작업 실행 진입점.

        1) 입력 이미지 목록 수집
        2) YOLO 모델 로드 및 클래스 이름 획득
        3) 각 이미지에 대해 추론/후처리/저장
        """
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
            shapes: List[dict[str, Any]] = []

            # 마스크가 존재할 때만 폴리곤 생성 로직 수행
            if (
                getattr(result, "masks", None) is not None
                and result.masks
                and result.masks.data is not None
            ):
                mask_count = int(result.masks.data.shape[0])

                cls_ids = (
                    result.boxes.cls.cpu().numpy().astype(int)
                    if (result.boxes is not None and result.boxes.cls is not None)
                    else np.zeros((mask_count,), dtype=int)
                )
                confs = (
                    result.boxes.conf.cpu().numpy()
                    if (result.boxes is not None and result.boxes.conf is not None)
                    else np.ones((mask_count,), dtype=float)
                )

                for i in range(mask_count):
                    class_id = int(cls_ids[i]) if i < len(cls_ids) else 0
                    score = float(confs[i]) if i < len(confs) else 1.0
                    label = (
                        class_names[class_id]
                        if 0 <= class_id < len(class_names)
                        else f"class_{class_id}"
                    )

                    polygons = get_ultra_segments(
                        result,
                        i,
                        approx_eps=float(self.config.approx_eps),
                    )
                    if not polygons:
                        continue

                    clipped = [clip_points(poly, width, height) for poly in polygons]
                    shapes.extend(
                        to_labelme_shapes(
                            clipped,
                            label,
                            class_id,
                            score,
                            float(self.config.min_area),
                        )
                    )

            if shapes:
                self._save_with_shapes(
                    image_path,
                    width,
                    height,
                    shapes,
                    index,
                    total,
                )
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
        """세그먼트가 없을 때 빈 yolo-seg txt와 선택적 시각화를 저장합니다."""
        bucket = "lt_0.60"
        txt_dir = self.config.save_root / bucket / "yolo-seg"
        viz_dir = self.config.save_root / bucket / "viz"
        txt_path = txt_dir / f"{image_path.stem}.txt"

        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("", encoding="utf-8")

        if self.config.copy_images:
            try:
                copy_file(
                    image_path,
                    txt_dir / image_path.name,
                    self.config.copy_mode,
                )
            except Exception as exc:
                self._emit(False, f"[{image_path.name}] 이미지 보관 실패: {exc}")

        if self.config.viz:
            img = cv2.imread(str(image_path))
            if img is not None:
                viz_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(viz_dir / image_path.name), img)

        progress = (
            f" ({index}/{total})"
            if index is not None and total is not None
            else ""
        )
        self._emit(
            True,
            f"[{image_path.name}] segments=0 -> bin={bucket} "
            f"-> SAVE empty .txt{progress}",
        )

    def _save_with_shapes(
        self,
        image_path: Path,
        width: int,
        height: int,
        shapes: Sequence[dict[str, Any]],
        index: int,
        total: int,
    ) -> None:
        """폴리곤 결과를 저장하고, 버킷 분류/시각화를 수행합니다."""
        scores = [
            float(shape.get("flags", {}).get("score", 1.0)) for shape in shapes
        ]
        aggregate = min(scores) if scores else 1.0
        bucket = score_to_bucket(aggregate)

        txt_dir = self.config.save_root / bucket / "yolo-seg"
        viz_dir = self.config.save_root / bucket / "viz"
        txt_path = txt_dir / f"{image_path.stem}.txt"

        save_yolo_seg_txt(txt_path, width, height, shapes)

        if self.config.copy_images:
            try:
                copy_file(
                    image_path,
                    txt_dir / image_path.name,
                    self.config.copy_mode,
                )
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
            f"[{image_path.name}] segments={len(shapes)} "
            f"score(min)={aggregate:.3f} -> bin={bucket} ({index}/{total})",
        )
