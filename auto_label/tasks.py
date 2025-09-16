import os
import threading
from PIL import Image
from PyQt5.QtCore import QRunnable

from .config import TARGET_SIZE, SAVE_JPEG_SPEED_PARAMS
from .utils import RESAMPLE
from .signals import Signals


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
            # 이미 처리되어 있으면 스킵
            if os.path.exists(self.out_path):
                self.signals.one_done.emit(True, f"[SKIP] exists: {os.path.basename(self.out_path)}")
                return

            # 파일 열기
            with Image.open(self.in_path) as img:
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

                # 최종 리사이즈 (종횡비 무시하고 정확히 1280x720)
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
