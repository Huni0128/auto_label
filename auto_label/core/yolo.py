"""YOLO specific helpers shared across services."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

BIN_LO = 0.60
BIN_STEP = 0.05
BIN_TOP = 0.95


def score_to_bucket(score: float) -> str:
    if score < BIN_LO:
        return "lt_0.60"
    if score >= BIN_TOP:
        return "ge_0.95"
    k = int(math.floor((score - BIN_LO) / BIN_STEP))
    start = BIN_LO + k * BIN_STEP
    end = start + BIN_STEP
    return f"{start:.2f}_{end:.2f}"


def to_labelme_shapes(
    polygons: Iterable[Sequence[Sequence[float]]],
    label: str,
    class_id: int,
    score: float,
    min_area: float,
) -> List[dict]:
    shapes: List[dict] = []
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


def save_yolo_seg_txt(path: Path, width: int, height: int, shapes: Sequence[dict]) -> None:
    lines: List[str] = []
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] = pts[:, 0] / float(width)
        pts[:, 1] = pts[:, 1] / float(height)
        pts = np.clip(pts, 0.0, 1.0)
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        class_id = int(shape.get("flags", {}).get("class_id", shape.get("group_id", 0)))
        lines.append(f"{class_id} {flat}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_viz(image_bgr: np.ndarray, shapes: Sequence[dict]) -> np.ndarray:
    visualised = image_bgr.copy()
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.polylines(visualised, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        score = float(shape.get("flags", {}).get("score", 0.0))
        label = f"{shape['label']} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 4, y)
        cv2.rectangle(visualised, (x, y - th - 4), (x + tw + 4, y), (0, 255, 0), -1)
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


def get_ultra_segments(result, index: int, approx_eps: float = 0.0) -> List[np.ndarray]:
    """Extract polygons from an ``ultralytics`` segmentation result."""
    segments = getattr(getattr(result, "masks", None), "xy", None)
    if segments is None:
        return []
    polys: List[np.ndarray] = []
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