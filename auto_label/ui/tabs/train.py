"""Controller for the YOLO training tab."""

from __future__ import annotations

import csv
import math
import re
import threading
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...core.config import (
    TARGET_SIZE,
    TRAIN_DEFAULT_BATCH,
    TRAIN_DEFAULT_BATCH_AUTO,
    TRAIN_DEFAULT_EPOCHS,
)
from ...qt.signals import Signals
from ...services.train import TrainConfig, TrainRunner
from .common import append_log


class LossPlotCanvas(FigureCanvasQTAgg):
    """Simple Matplotlib canvas to display YOLO loss curves."""

    def __init__(self, parent=None) -> None:
        figure = Figure(figsize=(5.0, 2.6), tight_layout=True)
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.ax = self.figure.add_subplot(111)
        self._configure_axes()

    def _configure_axes(self) -> None:
        self.ax.set_title("Loss 곡선")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, linestyle="--", alpha=0.3)

    def reset(self) -> None:
        self.ax.clear()
        self._configure_axes()
        self.draw_idle()

    def plot_losses(
        self,
        epochs: list[float],
        train_losses: list[float],
        val_losses: list[float],
    ) -> None:
        """Update the plot using the provided loss values."""
        self.reset()
        has_data = False
        if epochs and train_losses:
            self.ax.plot(epochs, train_losses, label="Train", color="#1f77b4")
            has_data = True
        if epochs and any(not math.isnan(v) for v in val_losses):
            self.ax.plot(epochs, val_losses, label="Val", color="#ff7f0e")
            has_data = True
        if has_data:
            self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.draw_idle()


class TrainTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.model_path: Path | None = None
        self.data_path: Path | None = None
        self.resume_path: Path | None = None
        self.last_run_dir: Path | None = None
        self.current_workdir: Path | None = None
        self.last_train_success: bool = False

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

        if hasattr(self.window, "chkTrainResume"):
            self.window.chkTrainResume.stateChanged.connect(self._toggle_resume_mode)
        if hasattr(self.window, "btnTrainResume"):
            self.window.btnTrainResume.clicked.connect(self._select_resume)

        self.loss_canvas: LossPlotCanvas | None = None
        if hasattr(self.window, "widgetTrainLoss"):
            self.loss_canvas = LossPlotCanvas(parent=self.window.widgetTrainLoss)
            if hasattr(self.window, "layoutTrainLoss"):
                self.window.layoutTrainLoss.addWidget(self.loss_canvas)
            else:
                layout = QVBoxLayout(self.window.widgetTrainLoss)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.loss_canvas)
            self.loss_canvas.reset()

        self._toggle_resume_mode()
        self._update_label()

    def _toggle_batch_auto(self) -> None:
        if hasattr(self.window, "chkTrainBatchAuto") and hasattr(self.window, "spinTrainBatch"):
            self.window.spinTrainBatch.setEnabled(not self.window.chkTrainBatchAuto.isChecked())

    def _is_resume_enabled(self) -> bool:
        return bool(
            hasattr(self.window, "chkTrainResume") and self.window.chkTrainResume.isChecked()
        )

    def _toggle_resume_mode(self) -> None:
        enabled = self._is_resume_enabled()
        if hasattr(self.window, "btnTrainResume"):
            can_enable = enabled and not self.window.btnTrainStop.isEnabled()
            self.window.btnTrainResume.setEnabled(can_enable)
        self._update_label()

    def _select_model(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self.window, "모델(.pt) 선택", "", "PyTorch Model (*.pt);;All Files (*)"
        )
        if not file_name:
            return
        self.model_path = Path(file_name)
        self._update_label()

    def _select_resume(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self.window,
            "체크포인트(.pt) 선택",
            "",
            "YOLO Checkpoint (*.pt);;All Files (*)",
        )
        if not file_name:
            return
        self.resume_path = Path(file_name)
        if self.model_path is None:
            self.model_path = self.resume_path
        self._update_label()

    def _select_data(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self.window, "dataset.yaml 선택", "", "YAML (*.yaml);;All Files (*)"
        )
        if not file_name:
            return
        self.data_path = Path(file_name)
        self._update_label()

    def _can_start(self) -> bool:
        if self.data_path is None:
            return False
        if self._is_resume_enabled():
            return bool(self.resume_path)
        return bool(self.model_path)

    def _update_label(self) -> None:
        model_txt = str(self.model_path) if self.model_path else "(모델 미선택)"
        data_txt = str(self.data_path) if self.data_path else "(dataset.yaml 미선택)"
        resume_txt = "사용 안 함"
        if self._is_resume_enabled():
            resume_txt = str(self.resume_path) if self.resume_path else "(체크포인트 미선택)"
        self.window.labelTrainPaths.setText(
            f"모델: {model_txt}\n데이터셋: {data_txt}\n체크포인트: {resume_txt}"
        )
        can_start = self._can_start()
        self.window.btnTrainStart.setEnabled(can_start and not self.window.btnTrainStop.isEnabled())

    def _toggle_ui(self, running: bool) -> None:
        self.window.btnTrainModel.setEnabled(not running)
        self.window.btnTrainData.setEnabled(not running)
        self.window.btnTrainStart.setEnabled((not running) and self._can_start())
        self.window.btnTrainStop.setEnabled(running)

        for attr in [
            "spinTrainW",
            "spinTrainH",
            "spinTrainEpochs",
            "spinTrainBatch",
            "chkTrainBatchAuto",
        ]:
            if hasattr(self.window, attr):
                widget = getattr(self.window, attr)
                if attr == "spinTrainBatch":
                    widget.setEnabled((not running) and (not self.window.chkTrainBatchAuto.isChecked()))
                elif attr == "chkTrainBatchAuto":
                    widget.setEnabled(not running)
                else:
                    widget.setEnabled(not running)

        if hasattr(self.window, "chkTrainResume"):
            self.window.chkTrainResume.setEnabled(not running)
        if hasattr(self.window, "btnTrainResume"):
            self.window.btnTrainResume.setEnabled((not running) and self._is_resume_enabled())

    def _run_training(self) -> None:
        resume_enabled = self._is_resume_enabled()
        resume_path = self.resume_path if resume_enabled else None

        model_path = self.model_path
        if resume_path and (model_path is None or resume_enabled):
            model_path = resume_path

        if resume_enabled and not resume_path:
            QMessageBox.warning(self.window, "경고", "Resume를 위한 체크포인트(.pt)를 선택하세요.")
            return
        if not (model_path and self.data_path):
            QMessageBox.warning(self.window, "경고", "모델(.pt)과 dataset.yaml을 선택하세요.")
            return

        self.stop_event.clear()
        self.window.textTrainLog.clear()
        self.last_run_dir = None
        self.current_workdir = self.data_path.resolve().parent
        self.last_train_success = False
        if self.loss_canvas:
            self.loss_canvas.reset()

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

        resume_desc = f", resume={resume_path}" if resume_path else ""
        
        append_log(
            self.window.textTrainLog,
            (
                "학습 시작: "
                f"model={model_path}, data={self.data_path}, "
                f"imgsz=({imgsz_w}x{imgsz_h}), epochs={epochs}, batch={batch_desc}{resume_desc}"
            ),
        )
        self._toggle_ui(True)

        config = TrainConfig(
            model_path=model_path,
            data_yaml=self.data_path,
            imgsz_w=imgsz_w,
            imgsz_h=imgsz_h,
            epochs=epochs,
            batch=batch,
            workdir=workdir,
            resume_from=resume_path,
        )
        runner = TrainRunner(config=config, stop_event=self.stop_event, signals=self.signals)
        self.pool.start(runner)

    def _stop_training(self) -> None:
        if not self.window.btnTrainStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textTrainLog, "학습 중지 요청을 보냈습니다...")

    def _extract_results_path(self, message: str) -> Path | None:
        match = re.search(r"Results saved to\s+(.+)$", message)
        if not match:
            return None
        raw_path = match.group(1).strip().rstrip(".")
        raw_path = raw_path.strip().strip("\"'")
        raw_path = raw_path.strip("`")
        raw_path = raw_path.strip("\\")
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            base = self.current_workdir or Path.cwd()
            path = (base / path).resolve()
        return path

    def _on_log(self, ok: bool, message: str) -> None:
        append_log(self.window.textTrainLog, message)

        if "Results saved to" in message:
            run_dir = self._extract_results_path(message)
            if run_dir:
                self.last_run_dir = run_dir
        if "exit=0" in message or "학습 완료" in message:
            self.last_train_success = True
        if "비정상 종료" in message or "오류" in message:
            self.last_train_success = False

    def _update_model_paths_from_run(self) -> None:
        if not self.last_run_dir:
            return
        weights_dir = self.last_run_dir / "weights"
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"
        updated = False
        if best_path.exists():
            self.model_path = best_path
            append_log(self.window.textTrainLog, f"[INFO] 최적 가중치: {best_path}")
            updated = True
        if last_path.exists():
            self.resume_path = last_path
            append_log(self.window.textTrainLog, f"[INFO] 재학습 체크포인트: {last_path}")
            updated = True
        if updated:
            self._update_label()

    def _collect_candidate_csvs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.last_run_dir:
            candidates.append(self.last_run_dir / "results.csv")
        if self.current_workdir:
            runs_root = self.current_workdir / "runs"
            if runs_root.exists():
                try:
                    for task_dir in runs_root.iterdir():
                        if not task_dir.is_dir():
                            continue
                        for run_dir in task_dir.iterdir():
                            if not run_dir.is_dir():
                                continue
                            candidates.append(run_dir / "results.csv")
                except OSError:
                    pass
        return candidates

    def _find_latest_results_csv(self) -> Path | None:
        candidates = [path for path in self._collect_candidate_csvs() if path.exists()]
        if not candidates:
            return None
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _sum_losses(row: dict[str, str], prefix: str) -> float | None:
        values: list[float] = []
        for key, value in row.items():
            if not key.startswith(prefix) or not key.endswith("loss"):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return float(sum(values))

    def _update_loss_plot(self) -> None:
        if not self.loss_canvas:
            return
        csv_path = self._find_latest_results_csv()
        if not csv_path:
            self.loss_canvas.reset()
            return

        epochs: list[int] = []
        train_losses: list[float] = []
        val_losses: list[float] = []
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    epoch_raw = row.get("epoch")
                    try:
                        epoch = int(epoch_raw) + 1 if epoch_raw is not None else len(epochs) + 1
                    except (TypeError, ValueError):
                        epoch = len(epochs) + 1
                    epochs.append(epoch)

                    train_loss = self._sum_losses(row, "train/")
                    val_loss = self._sum_losses(row, "val/")

                    train_losses.append(train_loss if train_loss is not None else math.nan)
                    val_losses.append(val_loss if val_loss is not None else math.nan)
        except OSError:
            self.loss_canvas.reset()
            return

        if not epochs:
            self.loss_canvas.reset()
            return
        self.loss_canvas.plot_losses(epochs, train_losses, val_losses)
    def _on_done(self) -> None:
        self._toggle_ui(False)
        append_log(self.window.textTrainLog, "학습 종료.")

        if self.last_train_success:
            self._update_model_paths_from_run()
        self._update_loss_plot()
        self._toggle_resume_mode()

        if self.stop_event.is_set():
            QMessageBox.information(self.window, "학습 중지됨", "사용자 요청으로 학습이 중지되었습니다.")