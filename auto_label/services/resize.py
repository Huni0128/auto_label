"""Image resizing worker used by the GUI thread pool."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QRunnable
from PIL import Image

from ..core.constants import TARGET_SIZE
from ..core.imaging import ensure_rgb, reduce_for_speed, save_image, resize_exact
from ..qt.signals import Signals


@dataclass(frozen=True)
class ResizeJob:
    """Represents a single resize task."""

    source: Path
    destination: Path


class ResizeImageTask(QRunnable):
    """Qt runnable that resizes an image on a background thread."""

    def __init__(self, job: ResizeJob, stop_event: threading.Event, signals: Signals) -> None:
        super().__init__()
        self.job = job
        self.stop_event = stop_event
        self.signals = signals

    # Helper -----------------------------------------------------------------
    def _emit(self, ok: bool, message: str) -> None:
        if self.signals:
            self.signals.one_done.emit(ok, message)

    # QRunnable ---------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
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

                image = reduce_for_speed(image, TARGET_SIZE)
                image = resize_exact(image, TARGET_SIZE)
                save_image(image, self.job.destination)

            self._emit(True, f"[OK] {self.job.destination.name}")
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(False, f"[FAIL] {self.job.source.name} - {exc}")