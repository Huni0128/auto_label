"""증강/오토 라벨링에서 공통으로 사용하는 폴리곤 유틸리티."""

from __future__ import annotations

from typing import Iterable

import numpy as np

Point = tuple[float, float]


def polygon_area(points: list[Point] | tuple[Point, ...]) -> float:
    """신호등 공식(shoelace)으로 폴리곤 면적을 계산합니다.

    Args:
        points: 시계/반시계 방향의 꼭짓점 목록.

    Returns:
        양수 면적 값. 3점 미만이면 0.0.
    """
    if len(points) < 3:
        return 0.0

    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def clip_point(point: Point, width: int, height: int) -> Point:
    """포인트가 이미지 프레임 내부에 있도록 클리핑합니다.

    Args:
        point: (x, y) 좌표.
        width: 이미지 너비.
        height: 이미지 높이.

    Returns:
        프레임 경계 내로 클리핑된 좌표.
    """
    x, y = point
    return (
        max(0.0, min(float(x), max(width - 1, 0))),
        max(0.0, min(float(y), max(height - 1, 0))),
    )


def sanitize_polygon(
    points: Iterable[Point],
    width: int,
    height: int,
    min_area: float,
) -> list[Point]:
    """폴리곤 포인트를 정리하고 최소 면적으로 필터링합니다.

    - NaN/Inf 제거
    - 프레임 내로 클리핑
    - 연속 중복 포인트 제거(1e-3 허용오차)
    - 최소 꼭짓점 수(3) 및 최소 면적 기준 적용

    Args:
        points: 입력 포인트 이터러블.
        width: 이미지 너비.
        height: 이미지 높이.
        min_area: 허용할 최소 면적.

    Returns:
        정리된 포인트 리스트. 조건 미달 시 빈 리스트.
    """
    clean: list[Point] = []
    for x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        cx, cy = clip_point((x, y), width, height)

        if not clean or abs(cx - clean[-1][0]) > 1e-3 or abs(cy - clean[-1][1]) > 1e-3:
            clean.append((cx, cy))

    if len(clean) < 3:
        return []

    if polygon_area(clean) < float(min_area):
        return []

    return clean


def flatten_polygons(
    polygons: list[list[Point]] | tuple[list[Point], ...]
) -> tuple[list[Point], list[tuple[int, int]]]:
    """키포인트 기반 증강을 위해 폴리곤들을 평탄화합니다.

    Args:
        polygons: 각 폴리곤은 (x, y) 포인트 리스트.

    Returns:
        (flat, layout)
          - flat: 모든 폴리곤을 이어 붙인 포인트 리스트.
          - layout: 각 폴리곤의 [start, end) 인덱스 구간 리스트.
    """
    flat: list[Point] = []
    layout: list[tuple[int, int]] = []

    cursor = 0
    for poly in polygons:
        start = cursor
        flat.extend(poly)
        cursor += len(poly)
        layout.append((start, cursor))

    return flat, layout
