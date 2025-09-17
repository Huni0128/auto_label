"""Common Qt signal definitions."""
from PyQt5.QtCore import QObject, pyqtSignal


class Signals(QObject):
    """Thread-safe signals emitted by background workers."""

    one_done = pyqtSignal(bool, str)
    all_done = pyqtSignal()