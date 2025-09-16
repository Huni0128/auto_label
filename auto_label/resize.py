import sys, os, threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QLabel, QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool
from PIL import Image

# ===================== 사용자 설정 =====================
TARGET_SIZE = (1280, 720)
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
MAX_THREADS_CAP = 32  # 너무 크게 올리면 I/O 경합 ↑
LOG_EVERY_N = 100     # 성공 로그는 N개마다 1번만 출력 (로그 폭주 방지)
SAVE_JPEG_SPEED_PARAMS = dict(quality=85, subsampling=1, optimize=False, progressive=False)
# =======================================================

def get_resample_lanczos():
    # Pillow 10.x: Image.Resampling.LANCZOS
    # 이전 버전 호환: Image.LANCZOS
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS

RESAMPLE = get_resample_lanczos()


# --------- 공용 신호 객체 ----------
class Signals(QObject):
    one_done = pyqtSignal(bool, str)   # (success, message)
    all_done = pyqtSignal()            # 전체 배치 종료


# --------- 단일 파일 변환 작업 ----------
class ResizeTask(QRunnable):
    def __init__(self, in_path: str, out_path: str, stop_event: threading.Event, signals: Signals):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.stop_event = stop_event
        self.signals = signals

    def run(self):
        # 중지 요청 시 빠른 반환
        if self.stop_event.is_set():
            self.signals.one_done.emit(False, f"[STOPPED] {os.path.basename(self.in_path)}")
            return

        try:
            # 이미 처리되어 있으면 스킵 (재실행 시 빠르게 통과)
            if os.path.exists(self.out_path):
                self.signals.one_done.emit(True, f"[SKIP] exists: {os.path.basename(self.out_path)}")
                return

            # 파일 열기
            with Image.open(self.in_path) as img:
                # 중지 체크
                if self.stop_event.is_set():
                    self.signals.one_done.emit(False, f"[STOPPED] {os.path.basename(self.in_path)}")
                    return

                # 모드 정리(팔레트, CMYK 등 → RGB)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # 거대 원본인 경우 2배수 다운샘플 후 최종 resize (속도 개선)
                w, h = img.size
                tw, th = TARGET_SIZE
                factor = 1
                while (w // (factor * 2)) > (tw * 2) and (h // (factor * 2)) > (th * 2):
                    factor *= 2
                if factor > 1:
                    img = img.reduce(factor)

                # 최종 리사이즈 (종횡비 무시하고 정확히 1280x720로 맞춤)
                img = img.resize(TARGET_SIZE, RESAMPLE)

                # 저장 포맷 결정(원본 확장자 유지)
                ext = os.path.splitext(self.out_path)[1].lower()
                fmt = None
                if ext in ('.jpg', '.jpeg'):
                    fmt = 'JPEG'
                elif ext == '.png':
                    fmt = 'PNG'
                elif ext == '.webp':
                    fmt = 'WEBP'

                # 저장 (JPEG은 속도 우선 옵션 적용)
                if fmt == 'JPEG':
                    img.save(self.out_path, format=fmt, **SAVE_JPEG_SPEED_PARAMS)
                else:
                    img.save(self.out_path, format=fmt if fmt else None)

            self.signals.one_done.emit(True, f"[OK] {os.path.basename(self.out_path)}")
        except Exception as e:
            self.signals.one_done.emit(False, f"[FAIL] {os.path.basename(self.in_path)} - {e}")


# --------- 메인 위젯 ----------
class ImageResizer(QWidget):
    def __init__(self):
        super().__init__()
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

        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("이미지 해상도 변환기 (1280x720) - 병렬/로그/진행률")
        self.setGeometry(250, 180, 760, 560)

        main = QVBoxLayout()

        # 상태 라벨
        self.label = QLabel("디렉토리를 선택하세요")
        self.label.setAlignment(Qt.AlignLeft)

        # 버튼 영역
        btns = QHBoxLayout()
        self.btn_select = QPushButton("디렉토리 선택")
        self.btn_select.clicked.connect(self._select_directory)

        self.btn_run = QPushButton("변환 실행(병렬)")
        self.btn_run.clicked.connect(self._run_parallel)
        self.btn_run.setEnabled(False)

        self.btn_stop = QPushButton("중지")
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_stop.setEnabled(False)

        btns.addWidget(self.btn_select)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_stop)

        # 진행률
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")

        # 스레드 정보 라벨
        self.thread_label = QLabel(f"스레드: {self.pool.maxThreadCount()}개 사용")
        self.thread_label.setAlignment(Qt.AlignRight)

        # 로그
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("변환 로그가 표시됩니다...")

        main.addWidget(self.label)
        main.addLayout(btns)
        main.addWidget(self.progress)
        main.addWidget(self.thread_label)
        main.addWidget(self.log)
        self.setLayout(main)

    # ----------------- UI 편의 메서드 -----------------
    def _append_log(self, text: str):
        self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _toggle_ui(self, running: bool):
        self.btn_select.setEnabled(not running)
        self.btn_run.setEnabled(not running and self.directory is not None)
        self.btn_stop.setEnabled(running)

    # ----------------- 이벤트 핸들러 -----------------
    def _select_directory(self):
        d = QFileDialog.getExistingDirectory(self, "디렉토리 선택")
        if not d:
            return
        self.directory = d
        self.out_dir = os.path.join(d,"../raw_images_1280x720")
        os.makedirs(self.out_dir, exist_ok=True)
        self.label.setText(f"입력: {self.directory}\n출력: {self.out_dir}")
        self.progress.setValue(0)
        self.log.clear()
        self._append_log("준비 완료. '변환 실행(병렬)'을 눌러 시작하세요.")
        self.btn_run.setEnabled(True)

    def _run_parallel(self):
        if not self.directory:
            QMessageBox.warning(self, "경고", "디렉토리를 먼저 선택하세요.")
            return

        # 중지 플래그 초기화
        self.stop_event.clear()

        # 파일 수집 (하위 폴더 제외 / 필요 시 walk로 확장 가능)
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

        # QThreadPool은 큐가 비면 자동 정리됨. 전체 완료는 one_done 누적으로 판단.

    def _stop_all(self):
        if not self.btn_stop.isEnabled():
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
        self.progress.setValue(pct)

        if self.done >= self.total:
            self.signals.all_done.emit()

    def _on_all_done(self):
        self._toggle_ui(False)
        self.progress.setValue(100)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageResizer()
    w.show()
    sys.exit(app.exec_())
