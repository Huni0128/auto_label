"""Polygon helpers shared between augmentation and auto-labelling."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def clip_point(point: Point, width: int, height: int) -> Point:
    x, y = point
    return (
        max(0.0, min(float(x), max(width - 1, 0))),
        max(0.0, min(float(y), max(height - 1, 0))),
    )


def sanitize_polygon(points: Iterable[Point], width: int, height: int, min_area: float) -> List[Point]:
    clean: List[Point] = []
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


def flatten_polygons(polygons: Sequence[Sequence[Point]]) -> Tuple[List[Point], List[Tuple[int, int]]]:
    """Flatten polygons for keypoint-based augmentation."""
    flat: List[Point] = []
    layout: List[Tuple[int, int]] = []
    cursor = 0
    for poly in polygons:
        start = cursor
        flat.extend(poly)
        cursor += len(poly)
        layout.append((start, cursor))
    return flat, layout