"""LabelMe JSON augmentation worker.

GUI 기본값에 맞춰 Albumentations 파이프라인으로 LabelMe 샘플을 증강하고,
증강된 이미지/폴리곤을 기반으로 새로운 LabelMe JSON을 생성합니다.
"""

from __future__ import annotations

import base64
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
from PyQt5.QtCore import QRunnable

from ..core.config import AUGMENT_RESOLUTION, IMG_EXTS
from ..core.polygons import flatten_polygons, sanitize_polygon
from ..qt.signals import Signals


@dataclass(frozen=True)
class AugmentationConfig:
    """증강 작업에 필요한 설정 값.

    Attributes:
        multiplier: 생성할 증강 샘플 개수(k).
        min_polygon_area: 유효한 폴리곤으로 인정할 최소 넓이.
        target_width: 증강 결과 이미지 너비.
        target_height: 증강 결과 이미지 높이.
    """
    multiplier: int
    min_polygon_area: float = 5.0
    target_width: int = AUGMENT_RESOLUTION[0]
    target_height: int = AUGMENT_RESOLUTION[1]

    @property
    def target_size(self) -> tuple[int, int]:
        """(width, height) 형태의 타깃 사이즈를 반환합니다."""
        return (self.target_width, self.target_height)


def build_transforms(config: AugmentationConfig) -> A.Compose:
    """GUI 기본값으로 설정된 Albumentations 파이프라인을 생성합니다.

    Args:
        config: 증강 설정.

    Returns:
        Albumentations Compose 파이프라인.
    """
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
            A.Resize(
                config.target_height,
                config.target_width,
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _decode_image_data(image_data: str) -> np.ndarray | None:
    """LabelMe JSON의 base64 이미지 데이터를 디코드합니다.

    실패 시 None을 반환합니다.

    Args:
        image_data: base64 인코딩된 이미지 문자열.

    Returns:
        BGR(OpenCV) 이미지 배열 또는 None.
    """
    try:
        data = base64.b64decode(image_data)
    except Exception:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _resolve_image_path(meta: dict[str, Any], json_path: Path, input_dir: Path) -> Path | None:
    """LabelMe 메타에서 원본 이미지 경로를 추정합니다.

    imagePath가 존재하면 우선 사용하고, 없으면 JSON 파일명과 동일한 스템에
    허용 확장자(IMG_EXTS)를 붙여 탐색합니다.

    Args:
        meta: LabelMe 메타(dict).
        json_path: 현재 JSON 파일 경로.
        input_dir: 입력 루트 디렉토리.

    Returns:
        발견된 이미지 경로 또는 None.
    """
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


class LabelMeAugmentationTask(QRunnable):
    """단일 LabelMe 샘플에 대해 증강본을 생성하는 Qt Runnable 작업."""

    def __init__(
        self,
        json_path: Path,
        input_dir: Path,
        output_dir: Path,
        config: AugmentationConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        """작업을 초기화합니다.

        Args:
            json_path: 원본 LabelMe JSON 경로.
            input_dir: 입력 루트 경로.
            output_dir: 출력 루트 경로.
            config: 증강 설정.
            stop_event: 중단 신호 이벤트.
            signals: 진행/완료 신호를 전달할 Qt Signals.
        """
        super().__init__()
        self.json_path = json_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.stop_event = stop_event
        self.signals = signals

    def _emit(self, ok: bool, message: str) -> None:
        """UI로 진행 상태를 전파합니다."""
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - Qt thread pool에서 실행
        """작업 실행 진입점.

        1) JSON/이미지 로드
        2) 폴리곤 플래튼 및 키포인트 구성
        3) 증강 파이프라인 적용(k회)
        4) 결과 이미지/JSON 저장 및 신호 전송
        """
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

            # LabelMe의 polygon만 수집(3점 미만 제외)
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

            for k in range(max(1, self.config.multiplier)):
                if self.stop_event.is_set():
                    self._emit(False, f"[STOPPED] {self.json_path.stem} (k={k + 1})")
                    break

                keypoints_in = flat_keypoints if flat_keypoints else [(0.0, 0.0)]
                transformed = pipeline(image=image, keypoints=keypoints_in)
                aug_image = transformed["image"]
                aug_keypoints = transformed["keypoints"]
                height, width = aug_image.shape[:2]

                # 원본 shapes를 복제한 뒤 증강된 좌표로 갱신
                new_shapes = deepcopy(shapes)

                if flat_keypoints and len(aug_keypoints) >= 1:
                    keypoints_np = np.array(aug_keypoints, dtype=np.float32)

                    for layout_idx, (start, end) in enumerate(layout):
                        pts_aug = [
                            (float(keypoints_np[i, 0]), float(keypoints_np[i, 1]))
                            for i in range(start, end)
                        ]
                        pts_aug = sanitize_polygon(
                            pts_aug,
                            width,
                            height,
                            self.config.min_polygon_area,
                        )
                        target_idx = polygon_indices[layout_idx]

                        if len(pts_aug) >= 3:
                            new_shapes[target_idx]["points"] = [
                                [float(x), float(y)] for x, y in pts_aug
                            ]
                            new_shapes[target_idx]["shape_type"] = "polygon"
                        else:
                            # 증강 결과 유효하지 않은 폴리곤은 포인트 비움
                            new_shapes[target_idx]["points"] = []

                    # 포인트가 3점 미만인 폴리곤은 제거
                    new_shapes = [
                        s
                        for s in new_shapes
                        if not (
                            s.get("shape_type", "polygon") == "polygon"
                            and len(s.get("points", [])) < 3
                        )
                    ]
                else:
                    # 폴리곤 키포인트가 없으면 폴리곤은 모두 제거
                    new_shapes = [
                        s for s in new_shapes if s.get("shape_type", "polygon") != "polygon"
                    ]

                output_image_name = f"{base_stem}_aug{k + 1}{output_ext}"
                output_json_name = f"{base_stem}_aug{k + 1}.json"

                self.output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.output_dir / output_image_name), aug_image)

                new_meta = deepcopy(meta)
                new_meta["imagePath"] = output_image_name
                new_meta["imageWidth"] = int(width)
                new_meta["imageHeight"] = int(height)
                new_meta["imageData"] = None  # 파일로 분리 저장
                new_meta["shapes"] = new_shapes

                (self.output_dir / output_json_name).write_text(
                    json.dumps(new_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                self._emit(True, f"[OK] {output_image_name}")

        except Exception as exc:  # pragma: no cover - 방어적 처리
            self._emit(False, f"[FAIL] {self.json_path.name} - {exc}")
