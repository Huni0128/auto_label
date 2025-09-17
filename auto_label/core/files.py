"""파일 시스템 관련 유틸리티 함수 모음."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List

from .constants import IMG_EXTS


def iter_image_files(path: Path) -> Iterable[Path]:
    """
    지정된 경로 내의 이미지 파일을 재귀적으로 탐색하여 반환합니다.

    Args:
        path (Path): 이미지 파일 또는 디렉토리 경로.

    Yields:
        Path: 발견된 이미지 파일 경로.

    Raises:
        FileNotFoundError: 입력 경로가 유효한 이미지 파일/폴더가 아닐 경우.
    """
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.suffix.lower() in IMG_EXTS:
                yield candidate
    elif path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path
    else:
        raise FileNotFoundError(f"이미지/폴더 경로 오류: {path}")


def list_image_files(path: Path) -> List[Path]:
    """
    지정된 경로 내의 이미지 파일을 정렬된 리스트로 반환합니다.

    Args:
        path (Path): 이미지 파일 또는 디렉토리 경로.

    Returns:
        List[Path]: 정렬된 이미지 파일 경로 리스트.
    """
    files = list(iter_image_files(path))
    files.sort()
    return files


def copy_file(src: Path, dest: Path, mode: str = "copy") -> None:
    """
    파일을 복사/링크/이동합니다.

    Args:
        src (Path): 원본 파일 경로.
        dest (Path): 대상 파일 경로.
        mode (str): 작업 모드.
            - "copy": 일반 복사 (기본값)
            - "hardlink": 하드 링크 생성
            - "symlink": 심볼릭 링크 생성 (실패 시 복사)
            - "move": 파일 이동

    Raises:
        ValueError: 지원하지 않는 모드가 지정된 경우.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "hardlink":
        os.link(src, dest)
    elif mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dest)
        except OSError:
            shutil.copy2(src, dest)
    elif mode == "move":
        shutil.move(src, dest)
    else:
        raise ValueError(f"지원하지 않는 copy mode: {mode}")
