"""File-system helpers."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List

from .constants import IMG_EXTS


def iter_image_files(path: Path) -> Iterable[Path]:
    """Yield image files inside ``path`` recursively."""
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.suffix.lower() in IMG_EXTS:
                yield candidate
    elif path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path
    else:
        raise FileNotFoundError(f"이미지/폴더 경로 오류: {path}")


def list_image_files(path: Path) -> List[Path]:
    files = list(iter_image_files(path))
    files.sort()
    return files


def copy_file(src: Path, dest: Path, mode: str = "copy") -> None:
    """Copy/link/move a file based on ``mode``."""
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
        raise ValueError(f"Unknown copy mode: {mode}")