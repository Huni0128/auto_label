"""Controller for the resize tab."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.constants import LOG_EVERY_N, VALID_EXTS
from ...qt.signals import Signals
from ...services.resize import ResizeImageTask, ResizeJob
from .common import ProgressTracker, append_log


class ResizeTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.directory: Path | None = None
        self.output_dir: Path | None = None

        self.progress = ProgressTracker()
        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        self.window.labelPath.setAlignment(Qt.AlignLeft)
        self.window.labelPath.setText("디렉토리를 선택하세요")
        self.window.threadLabel.setAlignment(Qt.AlignRight)
        self.window.threadLabel.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")
        self.window.progressBar.setRange(0, 100)
        self.window.progressBar.setValue(0)
        self.window.progressBar.setFormat("%p%")
        self.window.textLog.setReadOnly(True)
        self.window.textLog.setPlaceholderText("변환 로그가 표시됩니다...")

        self.window.btnSelect.clicked.connect(self._select_directory)
        self.window.btnRun.clicked.connect(self._run_parallel)
        self.window.btnStop.clicked.connect(self._stop_all)
        self.window.btnRun.setEnabled(False)
        self.window.btnStop.setEnabled(False)

    # UI helpers ---------------------------------------------------------------
    def _toggle_ui(self, running: bool) -> None:
        self.window.btnSelect.setEnabled(not running)
        self.window.btnRun.setEnabled((not running) and self.directory is not None)
        self.window.btnStop.setEnabled(running)

    def _select_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "디렉토리 선택")
        if not directory:
            return

        self.directory = Path(directory)
        self.output_dir = (self.directory / "../raw_images_1280x720").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window.labelPath.setText(
            f"입력: {self.directory}\n출력: {self.output_dir}"
        )
        self.window.progressBar.setValue(0)
        self.window.textLog.clear()
        append_log(self.window.textLog, "준비 완료. '변환 실행(병렬)'을 눌러 시작하세요.")
        self.window.btnRun.setEnabled(True)

    def _run_parallel(self) -> None:
        if not self.directory:
            QMessageBox.warning(self.window, "경고", "디렉토리를 먼저 선택하세요.")
            return

        self.stop_event.clear()
        files: list[ResizeJob] = []
        output_dir = self.output_dir
        if output_dir is None:
            return
        for name in os.listdir(self.directory):
            src = self.directory / name
            if src.is_file() and name.lower().endswith(VALID_EXTS):
                files.append(ResizeJob(src, output_dir / name))

        self.progress.reset(len(files))
        if self.progress.total == 0:
            append_log(self.window.textLog, "처리할 이미지가 없습니다.")
            return

        append_log(self.window.textLog, f"변환 시작 (총 {self.progress.total}개) ...")
        self._toggle_ui(True)

        for job in files:
            task = ResizeImageTask(job, self.stop_event, self.signals)
            self.pool.start(task)

    def _stop_all(self) -> None:
        if not self.window.btnStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textLog, "중지 요청을 보냈습니다... (진행 중인 파일은 마무리 후 종료)")

    # Signal callbacks --------------------------------------------------------
    def _on_one_done(self, ok: bool, message: str) -> None:
        self.progress.update(ok)
        if (not ok) or message.startswith("[SKIP]") or (self.progress.done <= 100) or (
            self.progress.done % LOG_EVERY_N == 0
        ):
            append_log(self.window.textLog, message)
        self.window.progressBar.setValue(self.progress.percent())

        if self.progress.done >= self.progress.total:
            self.signals.all_done.emit()

    def _on_all_done(self) -> None:
        self._toggle_ui(False)
        self.window.progressBar.setValue(100)
        append_log(
            self.window.textLog,
            f"완료: 총 {self.progress.total} | 성공 {self.progress.success} | 실패 {self.progress.failed}",
        )

        if self.stop_event.is_set():
            QMessageBox.information(
                self.window,
                "중지됨",
                f"사용자 요청으로 중지되었습니다.\n진행 결과: 성공 {self.progress.success} / 실패 {self.progress.failed} / 총 {self.progress.total}",
            )
        elif self.progress.failed == 0:
            QMessageBox.information(
                self.window,
                "완료",
                f"모든 이미지({self.progress.success}/{self.progress.total})가 변환되었습니다.",
            )
        else:
            QMessageBox.warning(
                self.window,
                "완료(일부 실패)",
                f"{self.progress.success}/{self.progress.total} 성공, {self.progress.failed} 실패. 로그를 확인하세요.",
            )