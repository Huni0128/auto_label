"""Thin wrapper around the ``yolo`` CLI to train segmentation models."""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyQt5.QtCore import QRunnable

from ..qt.signals import Signals


@dataclass(frozen=True)
class TrainConfig:
    model_path: Path
    data_yaml: Path
    imgsz_w: int
    imgsz_h: int
    epochs: int
    batch: int
    workdir: Path

    def build_command(self) -> list[str]:
        if self.imgsz_w != self.imgsz_h:
            imgsz = f"{self.imgsz_h},{self.imgsz_w}"
            rect_flag: Iterable[str] = ["rect=True"]
        else:
            imgsz = str(self.imgsz_w)
            rect_flag = []

        return [
            "yolo",
            "segment",
            "train",
            f"model={self.model_path}",
            f"data={self.data_yaml}",
            f"imgsz={imgsz}",
            f"epochs={self.epochs}",
            f"batch={self.batch}",
            *rect_flag,
        ]


class TrainRunner(QRunnable):
    """Execute a YOLO training command and stream the output to the GUI."""

    def __init__(self, config: TrainConfig, stop_event: threading.Event, signals: Signals) -> None:
        super().__init__()
        self.config = config
        self.stop_event = stop_event
        self.signals = signals
        self.process: subprocess.Popen[str] | None = None

    def _emit(self, ok: bool, message: str) -> None:
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
        try:
            command = self.config.build_command()
            self._emit(True, "[YOLO] " + " ".join(shlex.quote(part) for part in command))

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self.process = subprocess.Popen(
                command,
                cwd=str(self.config.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            assert self.process.stdout is not None  # for type checkers
            for line in self.process.stdout:
                if self.stop_event.is_set():
                    self._emit(False, "[YOLO] 중지 요청됨. 프로세스 종료 시도...")
                    try:
                        self.process.terminate()
                    except Exception:
                        pass
                    break
                line = line.rstrip("\n")
                if line:
                    self._emit(True, f"[YOLO] {line}")

            try:
                return_code = self.process.wait(timeout=10)
            except Exception:
                return_code = self.process.poll()

            if return_code == 0:
                self._emit(True, "[YOLO] 학습 완료 (exit=0)")
            else:
                if self.stop_event.is_set():
                    self._emit(False, "[YOLO] 학습이 사용자에 의해 중지되었습니다.")
                else:
                    self._emit(False, f"[YOLO] 비정상 종료 (exit={return_code})")
        except FileNotFoundError:
            self._emit(False, "[ERR] 'yolo' 명령을 찾을 수 없습니다. (pip install ultralytics)")
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(False, f"[ERR] 학습 실행 실패: {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()