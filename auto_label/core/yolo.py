"""YOLO 관련 공통 헬퍼들."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def score_to_bucket(score: float, threshold: float = 0.60) -> str:
    """신뢰도 점수를 지정된 임계값 기준 버킷 문자열로 변환합니다.

    Args:
        score: 0~1 범위 신뢰도 점수.
        threshold: 기준으로 사용할 confidence 임계값.

    Returns:
        점수가 임계값 이상이면 ``"high_<threshold>"``,
        그렇지 않으면 ``"low_<threshold>"`` 형태의 버킷 라벨.
    """
    threshold = float(threshold)
    suffix = f"{threshold:.2f}"
    return f"high_{suffix}" if score >= threshold else f"low_{suffix}"


def to_labelme_shapes(
    polygons: Iterable[Sequence[Sequence[float]]],
    label: str,
    class_id: int,
    score: float,
    min_area: float,
) -> list[dict]:
    """폴리곤들을 LabelMe shape 딕셔너리 형태로 변환합니다.

    Args:
        polygons: 각 폴리곤은 (N, 2) 좌표 시퀀스.
        label: 클래스 라벨 이름.
        class_id: 클래스 정수 ID.
        score: 신뢰도 점수.
        min_area: 최소 허용 면적(px^2).

    Returns:
        LabelMe shape 딕셔너리 리스트.
    """
    shapes: list[dict] = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        if len(pts) < 3:
            continue

        area = float(abs(cv2.contourArea(pts)))
        if area < min_area:
            continue

        shapes.append(
            {
                "label": label,
                "points": pts.astype(float).tolist(),
                "group_id": int(class_id),
                "shape_type": "polygon",
                "flags": {
                    "class_id": int(class_id),
                    "score": float(score),
                    "area_px2": area,
                },
            }
        )
    return shapes


def save_yolo_seg_txt(
    path: Path,
    width: int,
    height: int,
    shapes: Sequence[dict],
) -> None:
    """YOLO-SEG 폴리곤 라벨을 txt 형식으로 저장합니다.

    각 라인은 "class_id x1 y1 x2 y2 ..." (정규화 좌표) 형태입니다.

    Args:
        path: 출력 txt 경로.
        width: 원본 이미지 너비.
        height: 원본 이미지 높이.
        shapes: LabelMe 스타일 shape 목록.
    """
    lines: list[str] = []
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.float32)
        if pts.shape[0] < 3:
            continue

        pts[:, 0] = pts[:, 0] / float(width)
        pts[:, 1] = pts[:, 1] / float(height)
        pts = np.clip(pts, 0.0, 1.0)

        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        class_id = int(
            shape.get("flags", {}).get("class_id", shape.get("group_id", 0))
        )
        lines.append(f"{class_id} {flat}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_viz(image_bgr: np.ndarray, shapes: Sequence[dict]) -> np.ndarray:
    """폴리곤과 라벨/점수를 이미지에 시각화합니다.

    Args:
        image_bgr: 원본 BGR 이미지(OpenCV).
        shapes: LabelMe 스타일 shape 목록.

    Returns:
        시각화가 그려진 이미지 복사본(BGR).
    """
    visualised = image_bgr.copy()

    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) < 3:
            continue

        # 폴리곤 외곽선
        cv2.polylines(visualised, [pts], True, (0, 0, 255), 2)

        # 라벨/스코어 박스
        x, y = pts[0]
        score = float(shape.get("flags", {}).get("score", 0.0))
        label = f"{shape['label']} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        y = max(th + 4, y)
        cv2.rectangle(
            visualised,
            (x, y - th - 4),
            (x + tw + 4, y),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            visualised,
            label,
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return visualised


def get_ultra_segments(
    result,
    index: int,
    approx_eps: float = 0.0,
) -> list[np.ndarray]:
    """Ultralytics 결과에서 세그먼트 폴리곤을 추출합니다.

    Args:
        result: Ultralytics 예측 결과(Result 객체).
        index: 대상 인스턴스 인덱스.
        approx_eps: >0일 때 RDP 근사 에psilon. 0이면 생략.

    Returns:
        각 폴리곤을 (N, 2) float32 배열로 담은 리스트.
    """
    segments = getattr(getattr(result, "masks", None), "xy", None)
    if segments is None:
        return []

    polys: list[np.ndarray] = []
    raw = segments[index]
    if raw is None:
        return polys

    if isinstance(raw, np.ndarray):
        raw = [raw]

    for segment in raw:
        if segment is None or len(segment) < 3:
            continue

        contour = np.asarray(segment, dtype=np.float32).reshape(-1, 1, 2)
        if approx_eps > 0:
            contour = cv2.approxPolyDP(contour, approx_eps, True)
        polys.append(contour.reshape(-1, 2))

    return polys
