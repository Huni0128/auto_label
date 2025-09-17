"""GUI 스레드 풀에서 사용하는 이미지 리사이즈 워커."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QRunnable
from PIL import Image

from ..core.imaging import ensure_rgb, reduce_for_speed, resize_exact, save_image
from ..qt.signals import Signals


@dataclass(frozen=True)
class ResizeJob:
    """단일 리사이즈 작업을 표현합니다.

    Attributes:
        source: 입력 이미지 경로.
        destination: 출력 이미지 경로.
        size: (width, height) 타깃 크기.
    """

    source: Path
    destination: Path
    size: tuple[int, int]


class ResizeImageTask(QRunnable):
    """백그라운드 스레드에서 이미지를 리사이즈하는 Qt Runnable."""

    def __init__(
        self,
        job: ResizeJob,
        stop_event: threading.Event,
        signals: Signals,
    ) -> None:
        """작업 인스턴스를 초기화합니다."""
        super().__init__()
        self.job = job
        self.stop_event = stop_event
        self.signals = signals

    # Helper -----------------------------------------------------------------
    def _emit(self, ok: bool, message: str) -> None:
        """UI로 진행/결과 메시지를 전송합니다."""
        if self.signals:
            self.signals.one_done.emit(ok, message)

    # QRunnable ---------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - Qt thread pool에서 실행
        """작업 실행 진입점."""
        if self.stop_event.is_set():
            self._emit(False, f"[STOPPED] {self.job.source.name}")
            return

        try:
            if self.job.destination.exists():
                self._emit(True, f"[SKIP] exists: {self.job.destination.name}")
                return

            with Image.open(self.job.source) as image:
                image = ensure_rgb(image)

                if self.stop_event.is_set():
                    self._emit(False, f"[STOPPED] {self.job.source.name}")
                    return

                # 속도 최적화를 위한 다운스케일 후, 정확한 타깃 크기로 리사이즈
                image = reduce_for_speed(image, self.job.size)
                image = resize_exact(image, self.job.size)
                save_image(image, self.job.destination)

            self._emit(True, f"[OK] {self.job.destination.name}")

        except Exception as exc:  # pragma: no cover - 방어적 처리
            self._emit(False, f"[FAIL] {self.job.source.name} - {exc}")
