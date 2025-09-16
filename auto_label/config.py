from typing import Tuple

# ===================== 사용자 설정 =====================
TARGET_SIZE: Tuple[int, int] = (1280, 720)
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
MAX_THREADS_CAP = 32
LOG_EVERY_N = 100
SAVE_JPEG_SPEED_PARAMS = dict(quality=85, subsampling=1, optimize=False, progressive=False)
