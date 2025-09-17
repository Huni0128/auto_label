"""Controller for the augmentation tab."""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.constants import AUGMENT_RESOLUTION, LOG_EVERY_N
from ...qt.signals import Signals
from ...services.augment import AugmentationConfig, LabelmeAugmentationTask
from .common import ProgressTracker, append_log


class AugmentationTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.input_dir: Path | None = None
        self.output_dir: Path | None = None

        self.progress = ProgressTracker()
        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        self.window.labelAugPaths.setAlignment(Qt.AlignLeft)
        self.window.labelAugPaths.setText("입력/출력 폴더를 선택하세요")
        self.window.threadLabelAug.setAlignment(Qt.AlignRight)
        self.window.threadLabelAug.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")
        self.window.progressAug.setRange(0, 100)
        self.window.progressAug.setValue(0)
        self.window.progressAug.setFormat("%p%")
        self.window.textAugLog.setReadOnly(True)
        self.window.textAugLog.setPlaceholderText("증강 로그가 표시됩니다...")

        self.window.spinAugWidth.setValue(AUGMENT_RESOLUTION[0])
        self.window.spinAugHeight.setValue(AUGMENT_RESOLUTION[1])

        self.window.btnAugIn.clicked.connect(self._select_input_dir)
        self.window.btnAugOut.clicked.connect(self._select_output_dir)
        self.window.btnAugRun.clicked.connect(self._run_augmentation)
        self.window.btnAugStop.clicked.connect(self._stop_augmentation)
        self.window.btnAugRun.setEnabled(False)
        self.window.btnAugStop.setEnabled(False)

    def _toggle_ui(self, running: bool) -> None:
        self.window.btnAugIn.setEnabled(not running)
        self.window.btnAugOut.setEnabled(not running)
        can_run = (not running) and (self.input_dir is not None) and (self.output_dir is not None)
        self.window.btnAugRun.setEnabled(can_run)
        self.window.btnAugStop.setEnabled(running)
        self.window.spinAugWidth.setEnabled(not running)
        self.window.spinAugHeight.setEnabled(not running)

    def _select_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "입력 폴더 선택 (LabelMe JSON + 이미지)")
        if not directory:
            return
        self.input_dir = Path(directory)
        self._update_label()

    def _select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "출력 폴더 선택 (증강 결과 저장)")
        if not directory:
            return
        self.output_dir = Path(directory)
        self._update_label()

    def _update_label(self) -> None:
        in_txt = str(self.input_dir) if self.input_dir else "(미지정)"
        out_txt = str(self.output_dir) if self.output_dir else "(미지정)"
        self.window.labelAugPaths.setText(f"입력: {in_txt}\n출력: {out_txt}")
        can_run = bool(self.input_dir and self.output_dir)
        self.window.btnAugRun.setEnabled(can_run and not self.window.btnAugStop.isEnabled())

    def _run_augmentation(self) -> None:
        if not (self.input_dir and self.output_dir):
            QMessageBox.warning(self.window, "경고", "입력/출력 폴더를 먼저 선택하세요.")
            return

        self.stop_event.clear()
        self.window.textAugLog.clear()
        self.window.progressAug.setValue(0)

        json_files = sorted(self.input_dir.glob("*.json"))
        if not json_files:
            append_log(self.window.textAugLog, "JSON 파일을 찾지 못했습니다.")
            return

        multiplier = max(1, int(self.window.spinMultiplier.value()))
        target_width = int(self.window.spinAugWidth.value())
        target_height = int(self.window.spinAugHeight.value())
        self.progress.reset(len(json_files) * multiplier)
        append_log(
            self.window.textAugLog,
            (
                "증강 시작: "
                f"JSON {len(json_files)}개, multiplier={multiplier}, "
                f"출력 해상도 {target_width}x{target_height} → 총 {self.progress.total} 샘플"
            ),
        )
        self._toggle_ui(True)

        config = AugmentationConfig(
            multiplier=multiplier,
            target_width=target_width,
            target_height=target_height,
        )
        for json_path in json_files:
            task = LabelmeAugmentationTask(
                json_path=json_path,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                config=config,
                stop_event=self.stop_event,
                signals=self.signals,
            )
            self.pool.start(task)

    def _stop_augmentation(self) -> None:
        if not self.window.btnAugStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textAugLog, "증강 중지 요청을 보냈습니다... (진행 중인 항목은 마무리 후 종료)")

    def _on_one_done(self, ok: bool, message: str) -> None:
        self.progress.update(ok)
        if (not ok) or message.startswith("[SKIP]") or (self.progress.done <= 100) or (
            self.progress.done % LOG_EVERY_N == 0
        ):
            append_log(self.window.textAugLog, message)
        self.window.progressAug.setValue(self.progress.percent())

        if self.progress.done >= self.progress.total:
            self.signals.all_done.emit()

    def _on_all_done(self) -> None:
        self._toggle_ui(False)
        self.window.progressAug.setValue(100)
        append_log(
            self.window.textAugLog,
            f"증강 완료: 총 {self.progress.total} | 성공 {self.progress.success} | 실패 {self.progress.failed}",
        )

        if self.stop_event.is_set():
            QMessageBox.information(
                self.window,
                "증강 중지됨",
                f"사용자 요청으로 중지되었습니다.\n결과: 성공 {self.progress.success} / 실패 {self.progress.failed} / 총 {self.progress.total}",
            )
        elif self.progress.failed == 0:
            QMessageBox.information(
                self.window,
                "증강 완료",
                f"모든 샘플({self.progress.success}/{self.progress.total}) 생성 완료.",
            )
        else:
            QMessageBox.warning(
                self.window,
                "증강 완료(일부 실패)",
                f"{self.progress.success}/{self.progress.total} 성공, {self.progress.failed} 실패. 로그를 확인하세요.",
            )