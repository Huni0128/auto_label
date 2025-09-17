"""Utilities for splitting a verified dataset into YOLO train/val splits."""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PyQt5.QtCore import QRunnable

from ..core.files import list_image_files
from ..qt.signals import Signals

# Dataset directories that represent split prefixes and should be ignored when
# computing the relative path of a sample.
_SPLIT_PREFIXES = {"train", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class DatasetEntry:
    """Represents an image/label pair inside a dataset."""

    image: Path
    label: Path
    relative: Path


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Configuration options for dataset splitting."""

    val_ratio: float = 0.2
    seed: int | None = None


def _strip_split_prefix(relative: Path) -> Path:
    """Remove known split prefixes (train/val/test) from ``relative``."""

    parts = relative.parts
    if not parts:
        return relative

    first = parts[0].lower()
    if first in _SPLIT_PREFIXES:
        if len(parts) == 1:
            return Path(relative.name)
        return Path(*parts[1:])
    return relative


def _relative_key(relative: Path) -> str:
    """Return a key that uniquely identifies a sample without its suffix."""

    parent = relative.parent
    stem = relative.stem
    return str(parent / stem) if parent != Path(".") else stem


def collect_dataset_items(
    dataset_root: Path,
) -> Tuple[List[DatasetEntry], List[str], List[str], List[str]]:
    """Collect dataset entries and detect potential issues.

    Returns a tuple ``(items, missing_labels, missing_images, duplicates)`` where:

    * ``items`` – valid image/label pairs found in the dataset.
    * ``missing_labels`` – images that do not have a corresponding label file.
    * ``missing_images`` – label files that do not have a matching image.
    * ``duplicates`` – images that map to the same relative output path.
    """

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    missing_dirs: List[str] = []
    if not images_root.exists():
        missing_dirs.append("images")
    if not labels_root.exists():
        missing_dirs.append("labels")
    if missing_dirs:
        joined = ", ".join(missing_dirs)
        raise FileNotFoundError(f"필수 폴더가 없습니다: {joined}")

    items: List[DatasetEntry] = []
    missing_labels: List[str] = []
    duplicates: List[str] = []

    entries_by_key: Dict[str, DatasetEntry] = {}

    for image_path in list_image_files(images_root):
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(relative.as_posix())
            continue

        normalized = _strip_split_prefix(relative)
        key = _relative_key(normalized)
        if key in entries_by_key:
            duplicates.append(relative.as_posix())
            continue

        entry = DatasetEntry(image=image_path, label=label_path, relative=normalized)
        entries_by_key[key] = entry
        items.append(entry)

    missing_images: List[str] = []
    for label_path in labels_root.rglob("*.txt"):
        relative = label_path.relative_to(labels_root)
        normalized = _strip_split_prefix(relative)
        key = _relative_key(normalized)
        if key not in entries_by_key:
            missing_images.append(relative.as_posix())

    items.sort(key=lambda entry: entry.relative.as_posix())
    return items, missing_labels, missing_images, duplicates


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _move_path(src: Path, dest: Path) -> None:
    if src == dest:
        return
    _ensure_parent(dest)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.move(str(src), str(dest))


def _cleanup_empty_dirs(root: Path, preserve: Iterable[str] = ("train", "val")) -> None:
    if not root.exists():
        return

    preserve_lower = {name.lower() for name in preserve}
    # Iterate deepest directories first.
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(relative.parts) and relative.parts[0].lower() in preserve_lower:
            continue
        try:
            if not any(path.iterdir()):
                path.rmdir()
        except OSError:
            continue


def _copy_file(src: Path, dest: Path) -> None:
    _ensure_parent(dest)
    shutil.copy2(src, dest)


def _normalize(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


class DatasetSplitRunner(QRunnable):
    """Background worker that organises samples into train/val subdirectories."""

    def __init__(
        self,
        dataset_root: Path,
        items: Sequence[DatasetEntry],
        config: DatasetSplitConfig,
        signals: Signals | None = None,
        output_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.output_root = output_root or dataset_root
        self.items = list(items)
        self.config = config
        self.signals = signals

        self.copy_mode = _normalize(self.output_root) != _normalize(self.dataset_root)

        self.result_counts: Dict[str, int] = {"total": 0, "train": 0, "val": 0}
        self.errors: int = 0
        self.skipped: int = 0
        self.requested_total: int = 0

    def _emit(self, ok: bool, message: str) -> None:
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - executed in Qt worker threads
        try:
            total = len(self.items)
            self.requested_total = total
            self.result_counts = {"total": total, "train": 0, "val": 0}

            if total == 0:
                self._emit(False, "[WARN] 분할할 항목이 없습니다.")
                return

            ratio = max(0.0, min(1.0, float(self.config.val_ratio)))
            rng = random.Random(self.config.seed)
            entries = list(self.items)
            rng.shuffle(entries)

            val_count = min(total, int(round(total * ratio)))
            val_items = entries[:val_count]
            train_items = entries[val_count:]

            errors = 0
            for split_name, split_items in (("val", val_items), ("train", train_items)):
                for entry in split_items:
                    try:
                        image_dest = self.output_root / "images" / split_name / entry.relative
                        label_dest = self.output_root / "labels" / split_name / entry.relative.with_suffix(".txt")

                        if self.copy_mode:
                            if image_dest.exists() or label_dest.exists():
                                self.skipped += 1
                                self._emit(
                                    True,
                                    f"[WARN] {split_name.upper()} {entry.relative.as_posix()} 이미 존재하여 건너뜀",
                                )
                                self._emit(
                                    True,
                                    f"[{split_name.upper()}] {entry.relative.as_posix()} (건너뜀)",
                                )
                                continue
                            _copy_file(entry.image, image_dest)
                            _copy_file(entry.label, label_dest)
                        else:
                            _move_path(entry.image, image_dest)
                            _move_path(entry.label, label_dest)

                        self.result_counts[split_name] += 1
                        self._emit(True, f"[{split_name.upper()}] {entry.relative.as_posix()}")
                    except Exception as exc:  # pragma: no cover - defensive
                        errors += 1
                        self._emit(False, f"[ERR] {entry.relative.as_posix()} - {exc}")

            if not self.copy_mode:
                _cleanup_empty_dirs(self.dataset_root / "images")
                _cleanup_empty_dirs(self.dataset_root / "labels")

            self.errors = errors
            self.result_counts["total"] = self.result_counts["train"] + self.result_counts["val"]
            summary = f"[DONE] 학습 {self.result_counts['train']}건 · 검증 {self.result_counts['val']}건"
            if self.copy_mode and self.skipped:
                summary += f" (건너뜀 {self.skipped}건)"
            self._emit(errors == 0, summary)
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(False, f"[FAIL] 분할 중 오류: {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()