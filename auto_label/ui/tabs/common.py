"""Shared helpers for tab controllers."""
from __future__ import annotations

from dataclasses import dataclass
from PyQt5.QtWidgets import QTextEdit


@dataclass
class ProgressTracker:
    total: int = 0
    done: int = 0
    success: int = 0
    failed: int = 0

    def reset(self, total: int) -> None:
        self.total = total
        self.done = self.success = self.failed = 0

    def update(self, ok: bool) -> None:
        self.done += 1
        if ok:
            self.success += 1
        else:
            self.failed += 1

    def percent(self) -> int:
        if self.total <= 0:
            return 100
        return int(self.done * 100 / self.total)


def append_log(text_edit: QTextEdit, message: str) -> None:
    text_edit.append(message)
    text_edit.verticalScrollBar().setValue(text_edit.verticalScrollBar().maximum())