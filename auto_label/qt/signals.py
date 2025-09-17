"""Qt 스레드풀 작업자들이 공통으로 사용하는 시그널 정의 모듈."""

from PyQt5.QtCore import QObject, pyqtSignal


class Signals(QObject):
    """백그라운드 워커에서 스레드 세이프하게 발생시키는 공통 시그널."""

    one_done = pyqtSignal(bool, str)
    """개별 작업 단위 완료 시 (성공 여부, 메시지) 전달."""

    all_done = pyqtSignal()
    """모든 작업이 완료되었을 때 전달."""
