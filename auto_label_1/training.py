# training.py
# YOLO(seg) 학습을 서브프로세스로 실행하여 로그를 실시간 전달하고 중지 지원
import shlex
import subprocess
import threading
from pathlib import Path
from PyQt5.QtCore import QRunnable
from .signals import Signals

class TrainRunner(QRunnable):
    """
    yolo segment train model=... data=... imgsz=WxH 또는 S  epochs=... batch=...
    를 서브프로세스로 실행. stdout/stderr 라인을 GUI로 전달.
    """
    def __init__(
        self,
        model_path: Path,
        data_yaml: Path,
        imgsz_w: int,
        imgsz_h: int,
        epochs: int,
        batch,
        workdir: Path,
        signals: Signals,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.data_yaml = str(data_yaml)
        self.imgsz_w = int(imgsz_w)
        self.imgsz_h = int(imgsz_h)
        self.epochs = int(epochs)
        self.batch = batch
        self.workdir = str(workdir)
        self.signals = signals
        self.stop_event = stop_event
        self.proc = None

    def _emit(self, ok: bool, msg: str):
        if self.signals:
            self.signals.one_done.emit(ok, msg)

    def run(self):
        try:
            # batch 처리 동일
            if isinstance(self.batch, (int, float)):
                batch_arg = str(self.batch)
            else:
                batch_arg = "-1"

            # imgsz 처리 동일
            if self.imgsz_w != self.imgsz_h:
                imgsz_arg = f"{self.imgsz_h},{self.imgsz_w}"   # (H,W)
                rect_pair = ["rect=True"]
            else:
                imgsz_arg = str(self.imgsz_w)
                rect_pair = []

            cmd = [
                "yolo", "segment", "train",
                f"model={self.model_path}",
                f"data={self.data_yaml}",
                f"imgsz={imgsz_arg}",
                f"epochs={self.epochs}",
                f"batch={batch_arg}",
                *rect_pair,
            ]
            self._emit(True, "[YOLO] " + " ".join(shlex.quote(c) for c in cmd))

            import os
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self.proc = subprocess.Popen(
                cmd,
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            for line in self.proc.stdout:
                if self.stop_event.is_set():
                    self._emit(False, "[YOLO] 중지 요청됨. 프로세스 종료 시도...")
                    try:
                        self.proc.terminate()
                    except Exception:
                        pass
                    break
                line = line.rstrip("\n")
                if line:
                    self._emit(True, f"[YOLO] {line}")

            code = None
            try:
                code = self.proc.wait(timeout=10)
            except Exception:
                pass

            if code == 0:
                self._emit(True, "[YOLO] 학습 완료 (exit=0)")
            else:
                if self.stop_event.is_set():
                    self._emit(False, "[YOLO] 학습이 사용자에 의해 중지되었습니다.")
                else:
                    self._emit(False, f"[YOLO] 비정상 종료 (exit={code})")

        except FileNotFoundError:
            self._emit(False, "[ERR] 'yolo' 명령을 찾을 수 없습니다. (pip install ultralytics)")
        except Exception as e:
            self._emit(False, f"[ERR] 학습 실행 실패: {e}")
        finally:
            if self.signals:
                self.signals.all_done.emit()


