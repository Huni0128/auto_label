from PyQt5.QtCore import QObject, pyqtSignal


class Signals(QObject):
    one_done = pyqtSignal(bool, str)  # (success, message)
    all_done = pyqtSignal()           # 전체 배치 종료
