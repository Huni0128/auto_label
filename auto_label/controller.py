import os
import threading
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QThreadPool

from .config import VALID_EXTS, LOG_EVERY_N, MAX_THREADS_CAP
from .signals import Signals
from .tasks import ResizeTask
from .augment import AugmentTask  # 신규


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # UI 로드
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'main_window.ui')
        uic.loadUi(ui_path, self)

        # ========= 공통 상태 =========
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(MAX_THREADS_CAP, os.cpu_count() or 4))
        self.setWindowTitle("이미지 툴킷 - 리사이즈 & LabelMe 증강")
        self.resize(860, 640)

        # ========= 리사이즈 탭 =========
        self.directory = None
        self.out_dir = None
        self.total = self.done = self.success = self.failed = 0
        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        self.labelPath.setText("디렉토리를 선택하세요")
        self.labelPath.setAlignment(Qt.AlignLeft)
        self.threadLabel.setAlignment(Qt.AlignRight)
        self.threadLabel.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")

        self.btnSelect.clicked.connect(self._select_directory)
        self.btnRun.clicked.connect(self._run_parallel)
        self.btnStop.clicked.connect(self._stop_all)

        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(False)

        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("%p%")

        self.textLog.setReadOnly(True)
        self.textLog.setPlaceholderText("변환 로그가 표시됩니다...")

        # ========= 증강 탭 =========
        self.aug_in_dir = None
        self.aug_out_dir = None
        self.aug_total = self.aug_done = self.aug_success = self.aug_failed = 0
        self.aug_stop_event = threading.Event()
        self.aug_signals = Signals()
        self.aug_signals.one_done.connect(self._on_aug_one_done)
        self.aug_signals.all_done.connect(self._on_aug_all_done)

        self.labelAugPaths.setText("입력/출력 폴더를 선택하세요")
        self.labelAugPaths.setAlignment(Qt.AlignLeft)
        self.threadLabelAug.setAlignment(Qt.AlignRight)
        self.threadLabelAug.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")

        self.btnAugIn.clicked.connect(self._aug_select_in_dir)
        self.btnAugOut.clicked.connect(self._aug_select_out_dir)
        self.btnAugRun.clicked.connect(self._run_aug)
        self.btnAugStop.clicked.connect(self._stop_aug)

        self.btnAugRun.setEnabled(False)
        self.btnAugStop.setEnabled(False)

        self.progressAug.setRange(0, 100)
        self.progressAug.setValue(0)
        self.progressAug.setFormat("%p%")

        self.textAugLog.setReadOnly(True)
        self.textAugLog.setPlaceholderText("증강 로그가 표시됩니다...")

    # ----------------- 공용 유틸 -----------------
    def _append_log(self, text_edit, text: str):
        text_edit.append(text)
        text_edit.verticalScrollBar().setValue(text_edit.verticalScrollBar().maximum())

    # ----------------- 리사이즈 핸들러 -----------------
    def _toggle_ui(self, running: bool):
        self.btnSelect.setEnabled(not running)
        self.btnRun.setEnabled((not running) and (self.directory is not None))
        self.btnStop.setEnabled(running)

    def _select_directory(self):
        d = QFileDialog.getExistingDirectory(self, "디렉토리 선택")
        if not d:
            return
        self.directory = d
        self.out_dir = os.path.join(d, "../raw_images_1280x720")
        os.makedirs(self.out_dir, exist_ok=True)
        self.labelPath.setText(f"입력: {self.directory}\n출력: {os.path.abspath(self.out_dir)}")
        self.progressBar.setValue(0)
        self.textLog.clear()
        self._append_log(self.textLog, "준비 완료. '변환 실행(병렬)'을 눌러 시작하세요.")
        self.btnRun.setEnabled(True)

    def _run_parallel(self):
        if not self.directory:
            QMessageBox.warning(self, "경고", "디렉토리를 먼저 선택하세요.")
            return

        self.stop_event.clear()

        files = []
        for name in os.listdir(self.directory):
            in_path = os.path.join(self.directory, name)
            if os.path.isfile(in_path) and name.lower().endswith(VALID_EXTS):
                out_path = os.path.join(self.out_dir, name)  # 원본 확장자 유지
                files.append((in_path, out_path))

        self.total = len(files)
        self.done = self.success = self.failed = 0

        if self.total == 0:
            self._append_log(self.textLog, "처리할 이미지가 없습니다.")
            return

        self._append_log(self.textLog, f"변환 시작 (총 {self.total}개) ...")
        self._toggle_ui(True)

        for in_path, out_path in files:
            task = ResizeTask(in_path, out_path, self.stop_event, self.signals)
            self.pool.start(task)

    def _stop_all(self):
        if not self.btnStop.isEnabled():
            return
        self.stop_event.set()
        self._append_log(self.textLog, "중지 요청을 보냈습니다... (진행 중인 파일은 마무리 후 종료)")

    def _on_one_done(self, ok: bool, msg: str):
        self.done += 1
        if ok:
            self.success += 1
        else:
            self.failed += 1

        if (not ok) or msg.startswith("[SKIP]") or (self.done <= 100) or (self.done % LOG_EVERY_N == 0):
            self._append_log(self.textLog, msg)

        pct = int(self.done * 100 / self.total) if self.total else 100
        self.progressBar.setValue(pct)

        if self.done >= self.total:
            self.signals.all_done.emit()

    def _on_all_done(self):
        self._toggle_ui(False)
        self.progressBar.setValue(100)
        self._append_log(self.textLog, f"완료: 총 {self.total} | 성공 {self.success} | 실패 {self.failed}")

        if self.stop_event.is_set():
            QMessageBox.information(self, "중지됨",
                                    f"사용자 요청으로 중지되었습니다.\n진행 결과: 성공 {self.success} / 실패 {self.failed} / 총 {self.total}")
        elif self.failed == 0:
            QMessageBox.information(self, "완료",
                                    f"모든 이미지({self.success}/{self.total})가 변환되었습니다.")
        else:
            QMessageBox.warning(self, "완료(일부 실패)",
                                f"{self.success}/{self.total} 성공, {self.failed} 실패. 로그를 확인하세요.")

    # ----------------- 증강 핸들러 -----------------
    def _toggle_aug_ui(self, running: bool):
        self.btnAugIn.setEnabled(not running)
        self.btnAugOut.setEnabled(not running)
        self.btnAugRun.setEnabled((not running) and (self.aug_in_dir is not None) and (self.aug_out_dir is not None))
        self.btnAugStop.setEnabled(running)

    def _aug_select_in_dir(self):
        d = QFileDialog.getExistingDirectory(self, "입력 폴더 선택 (LabelMe JSON + 이미지)")
        if not d:
            return
        self.aug_in_dir = d
        self._update_aug_paths_label()

    def _aug_select_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "출력 폴더 선택 (증강 결과 저장)")
        if not d:
            return
        self.aug_out_dir = d
        self._update_aug_paths_label()

    def _update_aug_paths_label(self):
        in_txt = self.aug_in_dir if self.aug_in_dir else "(미지정)"
        out_txt = self.aug_out_dir if self.aug_out_dir else "(미지정)"
        self.labelAugPaths.setText(f"입력: {in_txt}\n출력: {out_txt}")
        self.btnAugRun.setEnabled(bool(self.aug_in_dir and self.aug_out_dir))

    def _run_aug(self):
        if not (self.aug_in_dir and self.aug_out_dir):
            QMessageBox.warning(self, "경고", "입력/출력 폴더를 먼저 선택하세요.")
            return

        self.aug_stop_event.clear()
        self.textAugLog.clear()
        self.progressAug.setValue(0)

        in_dir = Path(self.aug_in_dir).resolve()
        out_dir = Path(self.aug_out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted([p for p in in_dir.glob("*.json")])
        if len(json_files) == 0:
            self._append_log(self.textAugLog, "JSON 파일을 찾지 못했습니다.")
            return

        multiplier = max(1, int(self.spinMultiplier.value()))
        self.aug_total = len(json_files) * multiplier
        self.aug_done = self.aug_success = self.aug_failed = 0

        self._append_log(self.textAugLog, f"증강 시작: JSON {len(json_files)}개, multiplier={multiplier} → 총 {self.aug_total} 샘플")
        self._toggle_aug_ui(True)

        for jp in json_files:
            task = AugmentTask(
                json_path=jp,
                in_dir=in_dir,
                out_dir=out_dir,
                multiplier=multiplier,
                stop_event=self.aug_stop_event,
                signals=self.aug_signals,
            )
            self.pool.start(task)

    def _stop_aug(self):
        if not self.btnAugStop.isEnabled():
            return
        self.aug_stop_event.set()
        self._append_log(self.textAugLog, "증강 중지 요청을 보냈습니다... (진행 중인 항목은 마무리 후 종료)")

    def _on_aug_one_done(self, ok: bool, msg: str):
        self.aug_done += 1
        if ok:
            self.aug_success += 1
        else:
            self.aug_failed += 1

        if (not ok) or msg.startswith("[SKIP]") or (self.aug_done <= 100) or (self.aug_done % LOG_EVERY_N == 0):
            self._append_log(self.textAugLog, msg)

        pct = int(self.aug_done * 100 / self.aug_total) if self.aug_total else 100
        self.progressAug.setValue(pct)

        if self.aug_done >= self.aug_total:
            self.aug_signals.all_done.emit()

    def _on_aug_all_done(self):
        self._toggle_aug_ui(False)
        self.progressAug.setValue(100)
        self._append_log(self.textAugLog, f"증강 완료: 총 {self.aug_total} | 성공 {self.aug_success} | 실패 {self.aug_failed}")

        if self.aug_stop_event.is_set():
            QMessageBox.information(self, "증강 중지됨",
                                    f"사용자 요청으로 중지되었습니다.\n결과: 성공 {self.aug_success} / 실패 {self.aug_failed} / 총 {self.aug_total}")
        elif self.aug_failed == 0:
            QMessageBox.information(self, "증강 완료", f"모든 샘플({self.aug_success}/{self.aug_total}) 생성 완료.")
        else:
            QMessageBox.warning(self, "증강 완료(일부 실패)",
                                f"{self.aug_success}/{self.aug_total} 성공, {self.aug_failed} 실패. 로그를 확인하세요.")
