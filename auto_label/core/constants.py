"""Core constants shared across the application."""
from __future__ import annotations

from typing import Tuple

# === General image configuration ===
TARGET_SIZE: Tuple[int, int] = (1280, 720)
IMAGE_WIDTH, IMAGE_HEIGHT = TARGET_SIZE

# File extensions that are treated as images by the tooling.
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
IMG_EXTS = {ext.lower() for ext in VALID_EXTS if not ext.endswith("gif")} | {".gif"}

# Threading / logging configuration.
MAX_THREADS_CAP = 32
LOG_EVERY_N = 100

# Saving options that favour speed over compression for JPEG outputs.
SAVE_JPEG_SPEED_PARAMS = {
    "quality": 85,
    "subsampling": 1,
    "optimize": False,
    "progressive": False,
}

# Albumentations target resolution used by the augment task.
AUGMENT_RESOLUTION: Tuple[int, int] = TARGET_SIZE