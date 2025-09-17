"""애플리케이션 전반에서 사용하는 이미지 유틸리티."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from .config import SAVE_JPEG_SPEED_PARAMS


def get_resample_filter() -> int:
    """Pillow 9/10 모두에서 동작하는 고품질 리샘플링 필터를 반환합니다."""
    try:
        return Image.Resampling.LANCZOS  # Pillow >= 10
    except AttributeError:  # pragma: no cover - Pillow < 10
        return Image.LANCZOS


RESAMPLE = get_resample_filter()


def ensure_rgb(image: Image.Image) -> Image.Image:
    """후처리를 위해 이미지 색공간을 RGB(또는 RGBA)로 정규화합니다.

    Args:
        image: 입력 PIL 이미지.

    Returns:
        RGB 또는 RGBA 이미지. (이미 해당 모드면 그대로 반환)
    """
    if image.mode in {"RGB", "RGBA"}:
        return image
    return image.convert("RGB")


def reduce_for_speed(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """리사이즈 전 초대형 이미지를 2의 거듭제곱 스케일로 다운샘플합니다.

    매우 큰 이미지를 단계적으로 줄여 리사이즈 속도를 개선합니다.

    Args:
        image: 입력 PIL 이미지.
        target_size: (width, height) 타깃 크기.

    Returns:
        다운샘플된 이미지(필요 시) 또는 원본 이미지.
    """
    width, height = image.size
    target_w, target_h = target_size

    factor = 1
    while (
        (width // (factor * 2)) > (target_w * 2)
        and (height // (factor * 2)) > (target_h * 2)
    ):
        factor *= 2

    if factor > 1:
        return image.reduce(factor)
    return image


def resize_exact(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """종횡비를 무시하고 정확히 지정 크기로 리사이즈합니다.

    Args:
        image: 입력 이미지.
        size: (width, height) 타깃 크기.

    Returns:
        리사이즈된 이미지.
    """
    return image.resize(size, RESAMPLE)


def save_image(image: Image.Image, destination: Path) -> None:
    """가능하면 원본 확장자에 맞춰 이미지를 저장합니다.

    JPEG일 경우 속도 최적화 파라미터를 적용합니다.

    Args:
        image: 저장할 이미지.
        destination: 출력 파일 경로(.jpg/.jpeg/.png/.webp 권장).
    """
    ext = destination.suffix.lower()
    fmt: str | None = None

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
    new_shape: tuple[int, int] | int,
    colour: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, int, int, tuple[int, int]]:
    """종횡비를 유지하며 `new_shape`에 맞도록 패딩(letterbox) 리사이즈합니다.

    Args:
        image: 입력 PIL 이미지.
        new_shape: 정수(정사각) 또는 (width, height).
        colour: 패딩 색상(BGR 아님, PIL이므로 RGB 튜플).

    Returns:
        (canvas, scale, pad_w, pad_h, (orig_w, orig_h))
    """
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


def clip_points(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """폴리곤 좌표가 항상 이미지 프레임 내부에 있도록 클리핑합니다.

    Args:
        points: (x, y) 좌표 시퀀스.
        width: 이미지 너비.
        height: 이미지 높이.

    Returns:
        (N, 2) float32 배열의 클리핑된 좌표.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
    return pts


def draw_polylines(
    image: np.ndarray,
    polygons: Iterable[np.ndarray],
    colour: tuple[int, int, int],
) -> np.ndarray:
    """폴리곤들을 이미지 복사본 위에 선으로 그립니다.

    Args:
        image: BGR 배열(OpenCV 이미지).
        polygons: 각 폴리곤은 (N, 2) 정점 배열.
        colour: 선 색상(BGR 튜플).

    Returns:
        폴리곤이 그려진 이미지 복사본.
    """
    output = image.copy()
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.polylines(output, [pts], True, colour, 2)
    return output
