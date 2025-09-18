"""YOLO 학습/검증 분할을 수행하는 유틸리티 및 Qt 러너."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PyQt5.QtCore import QRunnable

from ..core.config import DEFAULT_SPLIT_SEED, DEFAULT_VAL_RATIO
from ..core.files import list_image_files
from ..qt.signals import Signals

# 분할(prefix) 디렉터리 이름: 상대 경로 계산 시 무시
_SPLIT_PREFIXES = {"train", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class DatasetEntry:
    """데이터셋 내 이미지/라벨 쌍을 표현합니다.

    Attributes:
        image: 이미지 파일 경로.
        label: 라벨(txt) 파일 경로.
        relative: 출력 하위경로 기준의 상대 경로.
    """

    image: Path
    label: Path
    relative: Path


@dataclass(frozen=True)
class DatasetSplitConfig:
    """데이터셋 분할 설정 값.

    Attributes:
        val_ratio: 검증 세트 비율(0~1).
        seed: 셔플 시드(None이면 비결정적).
    """

    val_ratio: float = DEFAULT_VAL_RATIO
    seed: int | None = DEFAULT_SPLIT_SEED


def _strip_split_prefix(relative: Path) -> Path:
    """known prefix(train/val/test 등)을 제거한 상대 경로를 반환합니다."""
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
    """확장자를 제외하고 샘플을 유일하게 식별하는 키를 생성합니다."""
    parent = relative.parent
    stem = relative.stem
    return str(parent / stem) if parent != Path(".") else stem


def collect_dataset_items(
    dataset_root: Path,
) -> tuple[list[DatasetEntry], list[str], list[str], list[str]]:
    """이미지/라벨 쌍을 수집하고 문제 파일을 탐지합니다.

    Returns:
        (items, missing_labels, missing_images, duplicates)

        - items: 유효한 이미지/라벨 쌍
        - missing_labels: 라벨이 없는 이미지 목록
        - missing_images: 이미지가 없는 라벨 목록
        - duplicates: 같은 상대 경로로 충돌하는 이미지 목록
    """
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    missing_dirs: list[str] = []
    if not images_root.exists():
        missing_dirs.append("images")
    if not labels_root.exists():
        missing_dirs.append("labels")
    if missing_dirs:
        joined = ", ".join(missing_dirs)
        raise FileNotFoundError(f"필수 폴더가 없습니다: {joined}")

    items: list[DatasetEntry] = []
    missing_labels: list[str] = []
    duplicates: list[str] = []

    entries_by_key: dict[str, DatasetEntry] = {}

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

        entry = DatasetEntry(
            image=image_path,
            label=label_path,
            relative=normalized,
        )
        entries_by_key[key] = entry
        items.append(entry)

    missing_images: list[str] = []
    for label_path in labels_root.rglob("*.txt"):
        relative = label_path.relative_to(labels_root)
        normalized = _strip_split_prefix(relative)
        key = _relative_key(normalized)
        if key not in entries_by_key:
            missing_images.append(relative.as_posix())

    items.sort(key=lambda e: e.relative.as_posix())
    return items, missing_labels, missing_images, duplicates


def _ensure_parent(path: Path) -> None:
    """대상 경로의 상위 디렉터리를 생성합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _move_path(src: Path, dest: Path) -> None:
    """파일/디렉터리를 이동합니다(기존 대상은 삭제)."""
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
    """루트 하위의 빈 디렉터리를 정리합니다(보존 디렉터리 제외)."""
    if not root.exists():
        return

    preserve_lower = {name.lower() for name in preserve}
    # 가장 깊은 경로부터 제거 시도
    for path in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
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
    """파일을 보존하여 복사합니다."""
    _ensure_parent(dest)
    shutil.copy2(src, dest)


def _normalize(path: Path) -> Path:
    """경로를 확장/정규화합니다."""
    return path.expanduser().resolve(strict=False)


class DatasetSplitRunner(QRunnable):
    """이미지/라벨 샘플을 train/val 하위 디렉터리로 분류하는 백그라운드 작업."""

    def __init__(
        self,
        dataset_root: Path,
        items: Sequence[DatasetEntry],
        config: DatasetSplitConfig,
        signals: Signals | None = None,
        output_root: Path | None = None,
    ) -> None:
        """작업 인스턴스를 초기화합니다."""
        super().__init__()
        self.dataset_root = dataset_root
        self.output_root = output_root or dataset_root
        self.items = list(items)
        self.config = config
        self.signals = signals

        self.copy_mode = (
            _normalize(self.output_root) != _normalize(self.dataset_root)
        )

        self.result_counts: dict[str, int] = {"total": 0, "train": 0, "val": 0}
        self.errors: int = 0
        self.skipped: int = 0
        self.requested_total: int = 0

    def _emit(self, ok: bool, message: str) -> None:
        """UI로 진행/결과 메시지를 전송합니다."""
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - Qt worker threads에서 실행
        """작업 실행 진입점."""
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
                        image_dest = (
                            self.output_root
                            / "images"
                            / split_name
                            / entry.relative
                        )
                        label_dest = (
                            self.output_root
                            / "labels"
                            / split_name
                            / entry.relative.with_suffix(".txt")
                        )

                        if self.copy_mode:
                            if image_dest.exists() or label_dest.exists():
                                self.skipped += 1
                                self._emit(
                                    True,
                                    f"[WARN] {split_name.upper()} "
                                    f"{entry.relative.as_posix()} 이미 존재하여 건너뜀",
                                )
                                self._emit(
                                    True,
                                    f"[{split_name.upper()}] "
                                    f"{entry.relative.as_posix()} (건너뜀)",
                                )
                                continue
                            _copy_file(entry.image, image_dest)
                            _copy_file(entry.label, label_dest)
                        else:
                            _move_path(entry.image, image_dest)
                            _move_path(entry.label, label_dest)

                        self.result_counts[split_name] += 1
                        self._emit(
                            True,
                            f"[{split_name.upper()}] {entry.relative.as_posix()}",
                        )
                    except Exception as exc:  # pragma: no cover - 방어적 처리
                        errors += 1
                        self._emit(
                            False,
                            f"[ERR] {entry.relative.as_posix()} - {exc}",
                        )

            if not self.copy_mode:
                _cleanup_empty_dirs(self.dataset_root / "images")
                _cleanup_empty_dirs(self.dataset_root / "labels")

            self.errors = errors
            self.result_counts["total"] = (
                self.result_counts["train"] + self.result_counts["val"]
            )
            summary = (
                f"[DONE] 학습 {self.result_counts['train']}건 · "
                f"검증 {self.result_counts['val']}건"
            )
            if self.copy_mode and self.skipped:
                summary += f" (건너뜀 {self.skipped}건)"
            self._emit(errors == 0, summary)

        except Exception as exc:  # pragma: no cover - 방어적 처리
            self._emit(False, f"[FAIL] 분할 중 오류: {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()
