"""Qt main window wiring for the application."""
from __future__ import annotations

import os
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QWidget

from ..core.config import MAX_THREADS_CAP, TARGET_SIZE
from .tabs.auto_label import AutoLabelTabController
from .tabs.augment import AugmentationTabController
from .tabs.convert import ConvertTabController
from .tabs.review import LabelReviewTabController
from .tabs.resize import ResizeTabController
from .tabs.split import DatasetSplitTabController
from .tabs.train import TrainTabController


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        ui_path = Path(__file__).resolve().parent / "main_window.ui"
        uic.loadUi(str(ui_path), self)

        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(MAX_THREADS_CAP, os.cpu_count() or 4))

        self.setWindowTitle("Auto Labeling PipeLine")
        self.resize(1000, 720)

        self._configure_defaults()

        self.resize_tab = ResizeTabController(self, self.pool)
        self.augment_tab = AugmentationTabController(self, self.pool)
        self.convert_tab = ConvertTabController(self, self.pool)
        self.train_tab = TrainTabController(self, self.pool)
        self.review_tab = LabelReviewTabController(self, self.pool)
        self.auto_label_tab = AutoLabelTabController(self, self.pool)
        self.split_tab = DatasetSplitTabController(self, self.pool)

    # Internal ----------------------------------------------------------------
    def _configure_defaults(self) -> None:
        default_w, default_h = TARGET_SIZE

        if hasattr(self, "spinResizeWidth"):
            self.spinResizeWidth.setValue(default_w)
        if hasattr(self, "spinResizeHeight"):
            self.spinResizeHeight.setValue(default_h)
        if hasattr(self, "spinCvW"):
            self.spinCvW.setValue(default_w)
        if hasattr(self, "spinCvH"):
            self.spinCvH.setValue(default_h)
        if hasattr(self, "doubleSpinValRatio"):
            self.doubleSpinValRatio.setValue(0.20)

        if hasattr(self, "spinAugWidth"):
            self.spinAugWidth.setValue(default_w)
        if hasattr(self, "spinAugHeight"):
            self.spinAugHeight.setValue(default_h)

        if hasattr(self, "spinTrainW"):
            self.spinTrainW.setValue(default_w)
        if hasattr(self, "spinTrainH"):
            self.spinTrainH.setValue(default_h)
        if hasattr(self, "spinTrainEpochs"):
            self.spinTrainEpochs.setValue(100)
        if hasattr(self, "spinTrainBatch"):
            self.spinTrainBatch.setValue(16)

        if hasattr(self, "spinALW"):
            self.spinALW.setValue(default_w)
        if hasattr(self, "spinALH"):
            self.spinALH.setValue(default_h)
        if hasattr(self, "doubleALConf"):
            self.doubleALConf.setValue(0.25)
        if hasattr(self, "doubleALIou"):
            self.doubleALIou.setValue(0.45)
        if hasattr(self, "doubleALApprox"):
            self.doubleALApprox.setValue(0.0)
        if hasattr(self, "doubleALMinArea"):
            self.doubleALMinArea.setValue(2000.0)
        if hasattr(self, "chkALViz"):
            self.chkALViz.setChecked(True)
        if hasattr(self, "chkALCopy"):
            self.chkALCopy.setChecked(True)
        if hasattr(self, "doubleSpinSplitVal"):
            self.doubleSpinSplitVal.setValue(0.20)
        if hasattr(self, "spinSplitSeed"):
            self.spinSplitSeed.setValue(42)