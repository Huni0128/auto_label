import os
import threading
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QThreadPool

from .config import VALID_EXTS, LOG_EVERY_N, MAX_THREADS_CAP
from .signals import Signals
from .tasks import ResizeTask


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # UI 로드
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'main_window.ui')
        uic.loadUi(ui_path, self)

        # 상태
        self.directory = None
        self.out_dir = None

        # 스레드 풀 (병렬 처리)
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(MAX_THREADS_CAP, os.cpu_count() or 4))

        # 진행 카운터
        self.total = 0
        self.done = 0
        self.success = 0
        self.failed = 0

        # 중지 이벤트
        self.stop_event = threading.Event()

        # 신호
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        # 위젯 핸들 (objectName 기준)
        self.labelPath.setText("디렉토리를 선택하세요")
        self.labelPath.setAlignment(Qt.AlignLeft)
        self.threadLabel.setAlignment(Qt.AlignRight)
        self.threadLabel.setText(f"스레드: {self.pool.maxThreadCount()}개 사용")

        # 버튼 연결
        self.btnSelect.clicked.connect(self._select_directory)
        self.btnRun.clicked.connect(self._run_parallel)
        self.btnStop.clicked.connect(self._stop_all)

        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(False)

        # 진행률
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("%p%")

        self.textLog.setReadOnly(True)
        self.textLog.setPlaceholderText("변환 로그가 표시됩니다...")

        self.setWindowTitle("이미지 해상도 변환기 (1280x720) - 모듈 구조")
        self.resize(760, 560)

    # ----------------- UI 편의 메서드 -----------------
    def _append_log(self, text: str):
        self.textLog.append(text)
        self.textLog.verticalScrollBar().setValue(self.textLog.verticalScrollBar().maximum())

    def _toggle_ui(self, running: bool):
        self.btnSelect.setEnabled(not running)
        self.btnRun.setEnabled((not running) and (self.directory is not None))
        self.btnStop.setEnabled(running)

    # ----------------- 이벤트 핸들러 -----------------
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
        self._append_log("준비 완료. '변환 실행(병렬)'을 눌러 시작하세요.")
        self.btnRun.setEnabled(True)

    def _run_parallel(self):
        if not self.directory:
            QMessageBox.warning(self, "경고", "디렉토리를 먼저 선택하세요.")
            return

        # 중지 플래그 초기화
        self.stop_event.clear()

        # 파일 수집 (하위 폴더 제외)
        files = []
        for name in os.listdir(self.directory):
            in_path = os.path.join(self.directory, name)
            if os.path.isfile(in_path) and name.lower().endswith(VALID_EXTS):
                out_path = os.path.join(self.out_dir, name)  # 원본 확장자 유지
                files.append((in_path, out_path))

        self.total = len(files)
        self.done = self.success = self.failed = 0

        if self.total == 0:
            self._append_log("처리할 이미지가 없습니다.")
            return

        self._append_log(f"변환 시작 (총 {self.total}개) ...")
        self._toggle_ui(True)

        # 작업 투입
        for in_path, out_path in files:
            task = ResizeTask(in_path, out_path, self.stop_event, self.signals)
            self.pool.start(task)

    def _stop_all(self):
        if not self.btnStop.isEnabled():
            return
        self.stop_event.set()
        self._append_log("중지 요청을 보냈습니다... (진행 중인 파일은 마무리 후 종료)")

    def _on_one_done(self, ok: bool, msg: str):
        self.done += 1
        if ok:
            self.success += 1
        else:
            self.failed += 1

        # 로그는 실패/스킵은 항상 출력, 성공은 처음 100개까지와 100의 배수마다만 출력
        if (not ok) or msg.startswith("[SKIP]") or (self.done <= 100) or (self.done % LOG_EVERY_N == 0):
            self._append_log(msg)

        pct = int(self.done * 100 / self.total) if self.total else 100
        self.progressBar.setValue(pct)

        if self.done >= self.total:
            self.signals.all_done.emit()

    def _on_all_done(self):
        self._toggle_ui(False)
        self.progressBar.setValue(100)
        self._append_log(f"완료: 총 {self.total} | 성공 {self.success} | 실패 {self.failed}")

        if self.stop_event.is_set():
            QMessageBox.information(self, "중지됨",
                                    f"사용자 요청으로 중지되었습니다.\n진행 결과: 성공 {self.success} / 실패 {self.failed} / 총 {self.total}")
        elif self.failed == 0:
            QMessageBox.information(self, "완료",
                                    f"모든 이미지({self.success}/{self.total})가 변환되었습니다.")
        else:
            QMessageBox.warning(self, "완료(일부 실패)",
                                f"{self.success}/{self.total} 성공, {self.failed} 실패. 로그를 확인하세요.")
