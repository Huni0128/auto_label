"""Controller for the dataset split tab."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.config import LOG_EVERY_N
from ...qt.signals import Signals
from ...services.split import (
    DatasetEntry,
    DatasetSplitConfig,
    DatasetSplitRunner,
    collect_dataset_items,
)
from .common import ProgressTracker, append_log


class DatasetSplitTabController:
    """UI controller that organises a dataset into train/val directories."""

    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.dataset_dir: Path | None = None
        self.output_dir: Path | None = None
        self.items: List[DatasetEntry] = []
        self.missing_labels: List[str] = []
        self.missing_images: List[str] = []
        self.duplicates: List[str] = []

        self.progress = ProgressTracker()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)
        self.current_runner: DatasetSplitRunner | None = None
        self.running: bool = False

        self.window.labelSplitPath.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.window.labelSplitOutput.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.window.labelSplitSummary.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.window.labelSplitPath.setText("분리할 데이터셋 폴더를 선택하세요.")
        self._set_output_dir(None)
        self.window.labelSplitSummary.setText("총 0건 · 학습 0건 · 검증 0건")
        self.window.progressSplit.setRange(0, 100)
        self.window.progressSplit.setValue(0)
        self.window.progressSplit.setFormat("%p%")
        self.window.textSplitLog.setReadOnly(True)
        self.window.textSplitLog.setPlaceholderText("분할 로그가 표시됩니다...")

        self.window.btnSplitSelect.clicked.connect(self._select_dataset)
        self.window.btnSplitOutput.clicked.connect(self._select_output_dataset)
        self.window.btnSplitRun.clicked.connect(self._run_split)
        self.window.btnSplitRun.setEnabled(False)
        if hasattr(self.window, "doubleSpinSplitVal"):
            self.window.doubleSpinSplitVal.valueChanged.connect(self._on_ratio_changed)

    # Helpers -----------------------------------------------------------------
    def _update_run_button(self) -> None:
        can_run = bool(self.dataset_dir and self.items and not self.running)
        self.window.btnSplitRun.setEnabled(can_run)

    def _set_dataset_dir(self, directory: Path) -> None:
        self.dataset_dir = directory
        self.window.labelSplitPath.setText(str(directory))
        self.window.textSplitLog.clear()
        append_log(self.window.textSplitLog, f"[DIR] 데이터셋 폴더: {directory}")
        if self.output_dir:
            append_log(self.window.textSplitLog, f"[OUT] 결과 데이터셋 폴더: {self.output_dir}")
        self._refresh_preview(log_warnings=True)

    def _set_output_dir(self, directory: Path | None) -> None:
        self.output_dir = directory
        if directory is None:
            self.window.labelSplitOutput.setText(
                "결과를 추가할 데이터셋 폴더를 선택하세요. (선택하지 않으면 입력 폴더 사용)"
            )
        else:
            self.window.labelSplitOutput.setText(str(directory))
            append_log(self.window.textSplitLog, f"[OUT] 결과 데이터셋 폴더: {directory}")

    def _refresh_preview(self, *, log_warnings: bool) -> None:
        if not self.dataset_dir:
            self.items = []
            self.missing_labels = []
            self.missing_images = []
            self.duplicates = []
            self.window.labelSplitSummary.setText("총 0건 · 학습 0건 · 검증 0건")
            self._update_run_button()
            return

        try:
            items, missing_labels, missing_images, duplicates = collect_dataset_items(self.dataset_dir)
        except FileNotFoundError as exc:
            QMessageBox.warning(self.window, "경고", str(exc))
            self.dataset_dir = None
            self.window.labelSplitPath.setText("분리할 데이터셋 폴더를 선택하세요.")
            self.window.labelSplitSummary.setText("총 0건 · 학습 0건 · 검증 0건")
            self.window.btnSplitRun.setEnabled(False)
            return

        self.items = items
        self.missing_labels = missing_labels
        self.missing_images = missing_images
        self.duplicates = duplicates

        self._update_summary()

        if log_warnings:
            self._log_dataset_warnings()

        self._update_run_button()

    def _log_dataset_warnings(self) -> None:
        if self.missing_labels:
            append_log(self.window.textSplitLog, self._format_list_warning("[WARN] 라벨 없음", self.missing_labels))
        if self.missing_images:
            append_log(self.window.textSplitLog, self._format_list_warning("[WARN] 이미지 없음", self.missing_images))
        if self.duplicates:
            append_log(self.window.textSplitLog, self._format_list_warning("[WARN] 중복 무시", self.duplicates))
        if not any([self.missing_labels, self.missing_images, self.duplicates]):
            append_log(self.window.textSplitLog, "[INFO] 데이터셋 검사가 완료되었습니다.")

    @staticmethod
    def _format_list_warning(prefix: str, items: List[str], limit: int = 3) -> str:
        sample = ", ".join(items[:limit])
        remainder = len(items) - limit
        if remainder > 0:
            sample = f"{sample} 외 {remainder}건"
        return f"{prefix}: {sample}"

    def _toggle_ui(self, running: bool) -> None:
        self.running = running
        self.window.btnSplitSelect.setEnabled(not running)
        self.window.btnSplitOutput.setEnabled(not running)
        self._update_run_button()

    def _update_summary(self) -> None:
        total = len(self.items)
        ratio = float(self.window.doubleSpinSplitVal.value()) if hasattr(self.window, "doubleSpinSplitVal") else 0.0
        val_estimate = min(total, int(round(total * ratio)))
        train_estimate = total - val_estimate

        lines = [f"총 {total}건 · 학습 예상 {train_estimate}건 · 검증 예상 {val_estimate}건"]
        if self.missing_labels:
            lines.append(f"라벨 없음 {len(self.missing_labels)}건")
        if self.missing_images:
            lines.append(f"이미지 없음 라벨 {len(self.missing_images)}건")
        if self.duplicates:
            lines.append(f"중복 이미지 {len(self.duplicates)}건 (무시됨)")
        self.window.labelSplitSummary.setText("\n".join(lines))

    # Slots -------------------------------------------------------------------
    def _select_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "데이터셋 폴더 선택")
        if not directory:
            return
        self._set_dataset_dir(Path(directory))

    def _select_output_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "결과 데이터셋 폴더 선택")
        if not directory:
            return
        self._set_output_dir(Path(directory))

    def _on_ratio_changed(self, _value: float) -> None:
        self._update_summary()

    def _run_split(self) -> None:
        if not self.dataset_dir:
            QMessageBox.warning(self.window, "경고", "데이터셋 폴더를 먼저 선택하세요.")
            return

        try:
            items, missing_labels, missing_images, duplicates = collect_dataset_items(self.dataset_dir)
        except FileNotFoundError as exc:
            QMessageBox.warning(self.window, "경고", str(exc))
            return

        if not items:
            QMessageBox.information(self.window, "알림", "분할할 이미지가 없습니다.")
            return

        warnings: List[str] = []
        if missing_labels:
            warnings.append(f"라벨 없음 {len(missing_labels)}건")
        if missing_images:
            warnings.append(f"이미지 없음 라벨 {len(missing_images)}건")
        if duplicates:
            warnings.append(f"중복 이미지 {len(duplicates)}건")

        if warnings:
            detail = "\n".join(warnings)
            reply = QMessageBox.question(
                self.window,
                "경고",
                f"다음 항목이 발견되었습니다:\n{detail}\n그래도 계속할까요?",
            )
            if reply != QMessageBox.Yes:
                return

        ratio = float(self.window.doubleSpinSplitVal.value())
        seed = int(self.window.spinSplitSeed.value())
        config = DatasetSplitConfig(val_ratio=ratio, seed=seed)

        output_dir = self.output_dir or self.dataset_dir
        try:
            same_output = output_dir.resolve() == self.dataset_dir.resolve()
        except Exception:
            same_output = output_dir == self.dataset_dir

        self.progress.reset(len(items))
        self.window.progressSplit.setValue(0)
        append_log(
            self.window.textSplitLog,
            f"[INFO] 분할 시작: 총 {len(items)}건, val_ratio={ratio:.2f}, seed={seed}",
        )
        if output_dir:
            mode = "입력과 동일 (이동)" if same_output else "기존 데이터셋에 추가"
            append_log(self.window.textSplitLog, f"[INFO] 결과 폴더: {output_dir} · {mode}")

        self._toggle_ui(True)

        runner = DatasetSplitRunner(self.dataset_dir, items, config, self.signals, output_root=output_dir)
        self.current_runner = runner
        self.pool.start(runner)

    def _on_one_done(self, ok: bool, message: str) -> None:
        if message.startswith("[TRAIN]") or message.startswith("[VAL]") or message.startswith("[ERR]"):
            self.progress.update(ok)
            self.window.progressSplit.setValue(self.progress.percent())

        if not message:
            return

        should_log = (not ok) or message.startswith("[ERR]") or message.startswith("[FAIL]")
        should_log |= message.startswith("[WARN]")
        if message.startswith("[TRAIN]") or message.startswith("[VAL]"):
            should_log |= self.progress.done <= 25 or (self.progress.done % LOG_EVERY_N == 0)
        if message.startswith("[DONE]"):
            should_log = True

        if should_log:
            append_log(self.window.textSplitLog, message)

    def _on_all_done(self) -> None:
        self._toggle_ui(False)
        if self.progress.total > 0:
            self.window.progressSplit.setValue(100)
        else:
            self.window.progressSplit.setValue(0)

        runner = self.current_runner
        counts = runner.result_counts if runner else {"total": 0, "train": 0, "val": 0}
        errors = runner.errors if runner else self.progress.failed
        requested_total = runner.requested_total if runner else self.progress.total
        skipped = runner.skipped if runner else 0
        self.current_runner = None

        # Refresh dataset summary without duplicating log warnings.
        self._refresh_preview(log_warnings=False)

        if requested_total > 0:
            if errors == 0:
                detail = f"학습 {counts['train']}건 · 검증 {counts['val']}건으로 분할했습니다."
                if skipped:
                    detail += f" (건너뜀 {skipped}건)"
                QMessageBox.information(
                    self.window,
                    "분할 완료",
                    detail,
                )
            else:
                QMessageBox.warning(
                    self.window,
                    "분할 완료 (일부 오류)",
                    f"총 {requested_total}건 중 {errors}건 실패했습니다. 로그를 확인하세요.",
                )
        elif errors:
            QMessageBox.warning(self.window, "분할 실패", "분할 중 오류가 발생했습니다. 로그를 확인하세요.")