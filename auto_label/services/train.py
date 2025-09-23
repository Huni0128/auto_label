"""Ultralytics `yolo` CLI를 얇게 감싼 세그멘테이션 학습 러너."""

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
    """YOLO 학습 실행에 필요한 설정 값.

    Attributes:
        model_path: YOLO 가중치(.pt) 경로.
        data_yaml: 학습 데이터셋 정의 YAML 경로.
        imgsz_w: 입력 리사이즈 가로.
        imgsz_h: 입력 리사이즈 세로.
        epochs: 학습 epoch 수.
        batch: 배치 크기.
        workdir: 프로세스 작업 디렉터리.
        resume_from: 이전 학습 체크포인트(.pt) 경로. ``None``이면 새로 학습.
    """

    model_path: Path
    data_yaml: Path
    imgsz_w: int
    imgsz_h: int
    epochs: int
    batch: int
    workdir: Path
    resume_from: Path | None = None

    def build_command(self) -> list[str]:
        """`yolo segment train` 명령 인자 리스트를 생성합니다."""
        if self.imgsz_w != self.imgsz_h:
            imgsz = f"{self.imgsz_h},{self.imgsz_w}"
            rect_flag: Iterable[str] = ["rect=True"]
        else:
            imgsz = str(self.imgsz_w)
            rect_flag = []

        command = [
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

        if self.resume_from:
            command.append(f"resume={self.resume_from}")

        return command

class TrainRunner(QRunnable):
    """YOLO 학습 명령을 실행하고 출력 스트림을 GUI로 전달하는 Qt Runnable."""

    def __init__(
        self,
        config: TrainConfig,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        """작업 인스턴스를 초기화합니다."""
        super().__init__()
        self.config = config
        self.stop_event = stop_event
        self.signals = signals
        self.process: subprocess.Popen[str] | None = None

    def _emit(self, ok: bool, message: str) -> None:
        """UI로 진행/결과 메시지를 전송합니다."""
        if self.signals:
            self.signals.one_done.emit(ok, message)

    def run(self) -> None:  # pragma: no cover - Qt thread pool에서 실행
        """작업 실행 진입점."""
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
        except Exception as exc:  # pragma: no cover - 방어적 처리
            self._emit(False, f"[ERR] 학습 실행 실패: {exc}")
        finally:
            if self.signals:
                self.signals.all_done.emit()
