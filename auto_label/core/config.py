"""애플리케이션 전역에서 공유하는 핵심 상수 모듈.

이 모듈은 이미지 처리, 로깅, 스레딩, 저장 옵션 등에 대한 공통 상수를
제공합니다. 팀 컨벤션(PEP 8)과 일관된 명명 규칙을 따릅니다.
"""

from __future__ import annotations

from typing import Final

# === 이미지 관련 공통 해상도 ===
TARGET_SIZE: Final[tuple[int, int]] = (1280, 720)

# 개별 축 크기를 명확히 사용해야 할 경우를 위한 별도 상수
IMAGE_WIDTH: Final[int] = TARGET_SIZE[0]
IMAGE_HEIGHT: Final[int] = TARGET_SIZE[1]

# === 이미지로 취급하는 파일 확장자 ===
# 툴링에서 이미지로 인식하는 확장자 목록 (소문자 사용 권장)
VALID_EXTS: Final[tuple[str, ...]] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
)

# 대소문자 혼용에 대비해 모두 소문자로 집합 구성.
# GIF는 애니메이션 여부와 무관하게 포함합니다.
IMG_EXTS: Final[set[str]] = {
    ext.lower() for ext in VALID_EXTS if not ext.endswith("gif")
} | {".gif"}

# === 데이터셋 기본 설정 ===
# 공통 검증 세트 비율 (0~1)
DEFAULT_VAL_RATIO: Final[float] = 0.20

# 데이터셋 분할 시 기본 난수 시드 (None이면 비결정적)
DEFAULT_SPLIT_SEED: Final[int | None] = 42


# === 스레딩 / 로깅 설정 ===
# 작업자 스레드 상한
MAX_THREADS_CAP: Final[int] = 32

# 진행 상황 로깅 간격 (N개마다 1회 로깅)
LOG_EVERY_N: Final[int] = 100

# === 저장 옵션 ===
# JPEG 저장 시 압축률보다 속도 우선 옵션.
# Pillow.save(**kwargs) 등에 그대로 전달해 사용합니다.
SAVE_JPEG_SPEED_PARAMS: Final[dict[str, int | bool]] = {
    "quality": 85,       # 시각적 품질과 용량의 균형값
    "subsampling": 1,    # 4:2:2
    "optimize": False,   # 최적화 비활성화(속도 우선)
    "progressive": False,
}

# === 증강(augmentation) 관련 ===
# Albumentations 등 증강 파이프라인의 기본 타깃 해상도
AUGMENT_RESOLUTION: Final[tuple[int, int]] = TARGET_SIZE

# 최소 폴리곤 면적 기본값 (LabelMe 증강 필터링 용도)
AUGMENT_MIN_POLYGON_AREA: Final[float] = 5.0

# === 학습 관련 기본값 ===
TRAIN_DEFAULT_EPOCHS: Final[int] = 100
TRAIN_DEFAULT_BATCH: Final[int] = 16
TRAIN_DEFAULT_BATCH_AUTO: Final[bool] = True

# === 오토 라벨링 기본값 ===
AUTO_LABEL_DEFAULT_CONF: Final[float] = 0.25
AUTO_LABEL_DEFAULT_IOU: Final[float] = 0.45
AUTO_LABEL_DEFAULT_APPROX_EPS: Final[float] = 0.0
AUTO_LABEL_DEFAULT_MIN_AREA: Final[float] = 2000.0
AUTO_LABEL_DEFAULT_VIZ: Final[bool] = True
AUTO_LABEL_DEFAULT_COPY_IMAGES: Final[bool] = True
AUTO_LABEL_DEFAULT_COPY_MODE: Final[str] = "copy"
AUTO_LABEL_COPY_MODES: Final[tuple[str, ...]] = (
    "copy",
    "hardlink",
    "symlink",
    "move",
)

# 기본 추론 디바이스(None이면 자동 선택)
AUTO_LABEL_DEFAULT_DEVICE: Final[str | None] = None