"""Imaging helpers used throughout the application."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .constants import SAVE_JPEG_SPEED_PARAMS


def get_resample_filter() -> int:
    """Return a high quality resampling filter that works on Pillow 9/10."""
    try:
        return Image.Resampling.LANCZOS  # Pillow >= 10
    except AttributeError:  # pragma: no cover - Pillow < 10
        return Image.LANCZOS


RESAMPLE = get_resample_filter()


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Normalise the colour space of an image for further processing."""
    if image.mode in {"RGB", "RGBA"}:
        return image
    return image.convert("RGB")


def reduce_for_speed(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Down-sample very large images by powers of two before resizing."""
    width, height = image.size
    target_w, target_h = target_size
    factor = 1
    while (width // (factor * 2)) > (target_w * 2) and (height // (factor * 2)) > (target_h * 2):
        factor *= 2
    if factor > 1:
        return image.reduce(factor)
    return image


def resize_exact(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize to the exact dimensions ignoring aspect ratio."""
    return image.resize(size, RESAMPLE)


def save_image(image: Image.Image, destination: Path) -> None:
    """Persist an image while preserving the original file type when possible."""
    ext = destination.suffix.lower()
    fmt = None
    if ext in {".jpg", ".jpeg"}:
        fmt = "JPEG"
    elif ext == ".png":
        fmt = "PNG"
    elif ext == ".webp":
        fmt = "WEBP"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "JPEG":
        image.save(destination, format=fmt, **SAVE_JPEG_SPEED_PARAMS)
    else:
        image.save(destination, format=fmt if fmt else None)


def letterbox(
    image: Image.Image,
    new_shape: Tuple[int, int] | int,
    colour: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, int, int, Tuple[int, int]]:
    """Resize with aspect ratio preserved and add padding to match ``new_shape``."""
    if isinstance(new_shape, int):
        new_w = new_h = int(new_shape)
    else:
        new_w, new_h = map(int, new_shape)

    width, height = image.size
    scale = min(new_w / width, new_h / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), colour)
    pad_w = (new_w - new_width) // 2
    pad_h = (new_h - new_height) // 2
    canvas.paste(resized, (pad_w, pad_h))

    return canvas, scale, pad_w, pad_h, (width, height)


def clip_points(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Clip polygon points so they always stay inside the image frame."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
    return pts


def draw_polylines(image: np.ndarray, polygons: Iterable[np.ndarray], colour: Tuple[int, int, int]) -> np.ndarray:
    """Draw polygons onto a copy of an image."""
    output = image.copy()
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.polylines(output, [pts], True, colour, 2)
    return output