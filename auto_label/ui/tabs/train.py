"""Controller for the YOLO training tab."""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.config import (
    TARGET_SIZE,
    TRAIN_DEFAULT_BATCH,
    TRAIN_DEFAULT_BATCH_AUTO,
    TRAIN_DEFAULT_EPOCHS,
)
from ...qt.signals import Signals
from ...services.train import TrainConfig, TrainRunner
from .common import append_log


class TrainTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.model_path: Path | None = None
        self.data_path: Path | None = None

        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_log)
        self.signals.all_done.connect(self._on_done)

        self.window.textTrainLog.setReadOnly(True)
        self.window.textTrainLog.setPlaceholderText("학습 로그가 표시됩니다...")

        self.window.btnTrainModel.clicked.connect(self._select_model)
        self.window.btnTrainData.clicked.connect(self._select_data)
        self.window.btnTrainStart.clicked.connect(self._run_training)
        self.window.btnTrainStop.clicked.connect(self._stop_training)
        self.window.btnTrainStart.setEnabled(False)
        self.window.btnTrainStop.setEnabled(False)

        if hasattr(self.window, "chkTrainBatchAuto"):
            self.window.chkTrainBatchAuto.setChecked(TRAIN_DEFAULT_BATCH_AUTO)
            self.window.chkTrainBatchAuto.stateChanged.connect(self._toggle_batch_auto)
            if hasattr(self.window, "spinTrainBatch"):
                self.window.spinTrainBatch.setEnabled(not TRAIN_DEFAULT_BATCH_AUTO)

    def _toggle_batch_auto(self) -> None:
        if hasattr(self.window, "chkTrainBatchAuto") and hasattr(self.window, "spinTrainBatch"):
            self.window.spinTrainBatch.setEnabled(not self.window.chkTrainBatchAuto.isChecked())

    def _select_model(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self.window, "모델(.pt) 선택", "", "PyTorch Model (*.pt);;All Files (*)")
        if not file_name:
            return
        self.model_path = Path(file_name)
        self._update_label()

    def _select_data(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self.window, "dataset.yaml 선택", "", "YAML (*.yaml);;All Files (*)")
        if not file_name:
            return
        self.data_path = Path(file_name)
        self._update_label()

    def _update_label(self) -> None:
        model_txt = str(self.model_path) if self.model_path else "(모델 미선택)"
        data_txt = str(self.data_path) if self.data_path else "(dataset.yaml 미선택)"
        self.window.labelTrainPaths.setText(f"모델: {model_txt}\n데이터셋: {data_txt}")
        can_start = bool(self.model_path and self.data_path)
        self.window.btnTrainStart.setEnabled(can_start and not self.window.btnTrainStop.isEnabled())

    def _toggle_ui(self, running: bool) -> None:
        self.window.btnTrainModel.setEnabled(not running)
        self.window.btnTrainData.setEnabled(not running)
        self.window.btnTrainStart.setEnabled((not running) and bool(self.model_path and self.data_path))
        self.window.btnTrainStop.setEnabled(running)

        for attr in ["spinTrainW", "spinTrainH", "spinTrainEpochs", "spinTrainBatch", "chkTrainBatchAuto"]:
            if hasattr(self.window, attr):
                widget = getattr(self.window, attr)
                if attr == "spinTrainBatch":
                    widget.setEnabled((not running) and (not self.window.chkTrainBatchAuto.isChecked()))
                elif attr == "chkTrainBatchAuto":
                    widget.setEnabled(not running)
                else:
                    widget.setEnabled(not running)

    def _run_training(self) -> None:
        if not (self.model_path and self.data_path):
            QMessageBox.warning(self.window, "경고", "모델(.pt)과 dataset.yaml을 선택하세요.")
            return

        self.stop_event.clear()
        self.window.textTrainLog.clear()

        imgsz_w = (
            int(self.window.spinTrainW.value())
            if hasattr(self.window, "spinTrainW")
            else TARGET_SIZE[0]
        )
        imgsz_h = (
            int(self.window.spinTrainH.value())
            if hasattr(self.window, "spinTrainH")
            else TARGET_SIZE[1]
        )
        epochs = (
            int(self.window.spinTrainEpochs.value())
            if hasattr(self.window, "spinTrainEpochs")
            else TRAIN_DEFAULT_EPOCHS
        )

        auto_batch_enabled = (
            hasattr(self.window, "chkTrainBatchAuto")
            and self.window.chkTrainBatchAuto.isChecked()
        )
        if not hasattr(self.window, "chkTrainBatchAuto"):
            auto_batch_enabled = TRAIN_DEFAULT_BATCH_AUTO

        if auto_batch_enabled:
            batch = -1
        else:
            batch = (
                int(self.window.spinTrainBatch.value())
                if hasattr(self.window, "spinTrainBatch")
                else TRAIN_DEFAULT_BATCH
            )

        workdir = self.data_path.resolve().parent

        batch_desc = "auto" if batch == -1 else batch
        
        append_log(
            self.window.textTrainLog,
            (
                "학습 시작: "
                f"model={self.model_path}, data={self.data_path}, "
                f"imgsz=({imgsz_w}x{imgsz_h}), epochs={epochs}, batch={batch_desc}"
            ),
        )
        self._toggle_ui(True)

        config = TrainConfig(
            model_path=self.model_path,
            data_yaml=self.data_path,
            imgsz_w=imgsz_w,
            imgsz_h=imgsz_h,
            epochs=epochs,
            batch=batch,
            workdir=workdir,
        )
        runner = TrainRunner(config=config, stop_event=self.stop_event, signals=self.signals)
        self.pool.start(runner)

    def _stop_training(self) -> None:
        if not self.window.btnTrainStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textTrainLog, "학습 중지 요청을 보냈습니다...")

    def _on_log(self, ok: bool, message: str) -> None:
        append_log(self.window.textTrainLog, message)

    def _on_done(self) -> None:
        self._toggle_ui(False)
        append_log(self.window.textTrainLog, "학습 종료.")
        if self.stop_event.is_set():
            QMessageBox.information(self.window, "학습 중지됨", "사용자 요청으로 학습이 중지되었습니다.")