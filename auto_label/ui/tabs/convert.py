"""Controller for the dataset conversion tab."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.config import LOG_EVERY_N
from ...qt.signals import Signals
from ...services.convert import ConvertConfig, DatasetConvertRunner
from .common import ProgressTracker, append_log


class ConvertTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.img_dir: Path | None = None
        self.json_dir: Path | None = None
        self.out_dir: Path | None = None

        self.progress = ProgressTracker()
        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        self.window.labelCvPaths.setAlignment(Qt.AlignLeft)
        self.window.labelCvPaths.setText("이미지/JSON/출력 폴더를 선택하세요")
        self.window.threadLabelCv.setAlignment(Qt.AlignRight)
        self.window.threadLabelCv.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")
        self.window.progressCv.setRange(0, 100)
        self.window.progressCv.setValue(0)
        self.window.progressCv.setFormat("%p%")
        self.window.textCvLog.setReadOnly(True)
        self.window.textCvLog.setPlaceholderText("변환 로그가 표시됩니다...")

        self.window.btnCvImg.clicked.connect(self._select_img_dir)
        self.window.btnCvJson.clicked.connect(self._select_json_dir)
        self.window.btnCvOut.clicked.connect(self._select_out_dir)
        self.window.btnCvRun.clicked.connect(self._run_conversion)
        self.window.btnCvStop.clicked.connect(self._stop_conversion)
        self.window.btnCvRun.setEnabled(False)
        self.window.btnCvStop.setEnabled(False)

    def _toggle_ui(self, running: bool) -> None:
        self.window.btnCvImg.setEnabled(not running)
        self.window.btnCvJson.setEnabled(not running)
        self.window.btnCvOut.setEnabled(not running)
        can_run = (not running) and all([self.img_dir, self.json_dir, self.out_dir])
        self.window.btnCvRun.setEnabled(can_run)
        self.window.btnCvStop.setEnabled(running)

    def _select_img_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "이미지 폴더 선택")
        if not directory:
            return
        self.img_dir = Path(directory)
        self._update_label()

    def _select_json_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "JSON 폴더 선택 (LabelMe/AnyLabeling 또는 COCO)")
        if not directory:
            return
        self.json_dir = Path(directory)
        self._update_label()

    def _select_out_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "출력 폴더 선택 (YOLO 데이터셋 생성)")
        if not directory:
            return
        self.out_dir = Path(directory)
        self._update_label()

    def _update_label(self) -> None:
        img_txt = str(self.img_dir) if self.img_dir else "(미지정)"
        json_txt = str(self.json_dir) if self.json_dir else "(미지정)"
        out_txt = str(self.out_dir) if self.out_dir else "(미지정)"
        self.window.labelCvPaths.setText(f"이미지: {img_txt}\nJSON: {json_txt}\n출력: {out_txt}")
        can_run = bool(self.img_dir and self.json_dir and self.out_dir)
        self.window.btnCvRun.setEnabled(can_run and not self.window.btnCvStop.isEnabled())

    def _estimate_total(self) -> int:
        if not self.json_dir:
            return 0
        for json_path in self.json_dir.glob("*.json"):
            try:
                obj = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if all(key in obj for key in ("images", "annotations", "categories")):
                return len(obj.get("images", []))
        return len(list(self.json_dir.glob("*.json")))

    def _run_conversion(self) -> None:
        if not all([self.img_dir, self.json_dir, self.out_dir]):
            QMessageBox.warning(self.window, "경고", "이미지/JSON/출력 폴더를 모두 선택하세요.")
            return

        self.stop_event.clear()
        self.window.textCvLog.clear()
        self.window.progressCv.setValue(0)

        total_estimate = max(1, self._estimate_total())
        self.progress.reset(total_estimate)

        w = int(self.window.spinCvW.value())
        h = int(self.window.spinCvH.value())
        val_ratio = float(self.window.doubleSpinValRatio.value())

        append_log(
            self.window.textCvLog,
            f"변환 시작: size=({w},{h}), val_ratio={val_ratio} → 예상 총 {self.progress.total} 샘플",
        )
        self._toggle_ui(True)

        config = ConvertConfig(size=(w, h), val_ratio=val_ratio)
        runner = DatasetConvertRunner(
            img_dir=self.img_dir,
            json_dir=self.json_dir,
            output_root=self.out_dir,
            config=config,
            stop_event=self.stop_event,
            signals=self.signals,
        )
        self.pool.start(runner)

    def _stop_conversion(self) -> None:
        if not self.window.btnCvStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textCvLog, "변환 중지 요청을 보냈습니다... (진행 중인 항목은 마무리 후 종료)")

    def _on_one_done(self, ok: bool, message: str) -> None:
        self.progress.update(ok)
        if (not ok) or (self.progress.done <= 100) or (self.progress.done % LOG_EVERY_N == 0):
            append_log(self.window.textCvLog, message)
        self.window.progressCv.setValue(self.progress.percent())

    def _on_all_done(self) -> None:
        self._toggle_ui(False)
        self.window.progressCv.setValue(100)
        append_log(
            self.window.textCvLog,
            f"변환 완료: 추정 총 {self.progress.total} | 성공 {self.progress.success} | 실패 {self.progress.failed}",
        )

        if self.stop_event.is_set():
            QMessageBox.information(
                self.window,
                "변환 중지됨",
                f"사용자 요청으로 중지되었습니다.\n결과: 성공 {self.progress.success} / 실패 {self.progress.failed}",
            )
        elif self.progress.failed == 0:
            QMessageBox.information(
                self.window,
                "변환 완료",
                f"모든 샘플 생성 완료. (성공 {self.progress.success})",
            )
        else:
            QMessageBox.warning(
                self.window,
                "변환 완료(일부 실패)",
                f"성공 {self.progress.success}, 실패 {self.progress.failed}. 로그를 확인하세요.",
            )