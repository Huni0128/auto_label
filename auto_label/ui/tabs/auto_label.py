"""Controller for the auto-labelling tab."""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...core.config import IMG_EXTS
from ...qt.signals import Signals
from ...services.auto_label import AutoLabelConfig, AutoLabelRunner
from .common import ProgressTracker, append_log


class AutoLabelTabController:
    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.model_path: Path | None = None
        self.image_dir: Path | None = None
        self.output_dir: Path | None = None

        self.progress = ProgressTracker()
        self.stop_event = threading.Event()
        self.signals = Signals()
        self.signals.one_done.connect(self._on_one_done)
        self.signals.all_done.connect(self._on_all_done)

        self.window.labelALPaths.setAlignment(Qt.AlignLeft)
        self.window.labelALPaths.setText("모델(.pt), 이미지 폴더, 저장 폴더를 선택하세요")
        self.window.progressAL.setRange(0, 100)
        self.window.progressAL.setValue(0)
        self.window.progressAL.setFormat("%p%")
        self.window.textALLog.setReadOnly(True)
        self.window.textALLog.setPlaceholderText("오토 라벨 로그가 표시됩니다...")

        self.window.btnALModel.clicked.connect(self._select_model)
        self.window.btnALImg.clicked.connect(self._select_image_dir)
        self.window.btnALSave.clicked.connect(self._select_output_dir)
        self.window.btnALRun.clicked.connect(self._run)
        self.window.btnALStop.clicked.connect(self._stop)
        self.window.btnALRun.setEnabled(False)
        self.window.btnALStop.setEnabled(False)

        if hasattr(self.window, "comboALCopyMode"):
            self.window.comboALCopyMode.addItems(["copy", "hardlink", "symlink", "move"])
            self.window.comboALCopyMode.setCurrentText("copy")

    def _toggle_ui(self, running: bool) -> None:
        self.window.btnALModel.setEnabled(not running)
        self.window.btnALImg.setEnabled(not running)
        self.window.btnALSave.setEnabled(not running)
        can_run = (not running) and bool(self.model_path and self.image_dir and self.output_dir)
        self.window.btnALRun.setEnabled(can_run)
        self.window.btnALStop.setEnabled(running)

        for attr in [
            "spinALW",
            "spinALH",
            "doubleALConf",
            "doubleALIou",
            "doubleALApprox",
            "doubleALMinArea",
            "chkALViz",
            "chkALCopy",
            "comboALCopyMode",
        ]:
            if hasattr(self.window, attr):
                getattr(self.window, attr).setEnabled(not running)

    def _select_model(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self.window, "모델(.pt) 선택", "", "PyTorch Model (*.pt);;All Files (*)")
        if not file_name:
            return
        self.model_path = Path(file_name)
        self._update_label()

    def _select_image_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "이미지 폴더 선택")
        if not directory:
            return
        self.image_dir = Path(directory)
        self._update_label()

    def _select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "저장 폴더 선택 (bin 하위로 저장)")
        if not directory:
            return
        self.output_dir = Path(directory)
        self._update_label()

    def _update_label(self) -> None:
        model_txt = str(self.model_path) if self.model_path else "(모델 미선택)"
        img_txt = str(self.image_dir) if self.image_dir else "(이미지 폴더 미선택)"
        save_txt = str(self.output_dir) if self.output_dir else "(저장 폴더 미선택)"
        self.window.labelALPaths.setText(f"모델: {model_txt}\n이미지: {img_txt}\n저장: {save_txt}")
        can_run = bool(self.model_path and self.image_dir and self.output_dir)
        self.window.btnALRun.setEnabled(can_run and not self.window.btnALStop.isEnabled())

    def _estimate_total(self) -> int:
        if not self.image_dir:
            return 0
        return sum(1 for p in self.image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)

    def _run(self) -> None:
        if not (self.model_path and self.image_dir and self.output_dir):
            QMessageBox.warning(self.window, "경고", "모델/이미지/저장 폴더를 모두 선택하세요.")
            return

        self.stop_event.clear()
        self.window.textALLog.clear()
        self.window.progressAL.setValue(0)

        imgsz_w = int(self.window.spinALW.value()) if hasattr(self.window, "spinALW") else 1280
        imgsz_h = int(self.window.spinALH.value()) if hasattr(self.window, "spinALH") else 720
        conf = float(self.window.doubleALConf.value()) if hasattr(self.window, "doubleALConf") else 0.25
        iou = float(self.window.doubleALIou.value()) if hasattr(self.window, "doubleALIou") else 0.45
        approx = float(self.window.doubleALApprox.value()) if hasattr(self.window, "doubleALApprox") else 0.0
        min_area = float(self.window.doubleALMinArea.value()) if hasattr(self.window, "doubleALMinArea") else 2000.0
        viz = bool(self.window.chkALViz.isChecked()) if hasattr(self.window, "chkALViz") else True
        copy_img = bool(self.window.chkALCopy.isChecked()) if hasattr(self.window, "chkALCopy") else True
        copy_mode = self.window.comboALCopyMode.currentText() if hasattr(self.window, "comboALCopyMode") else "copy"

        self.progress.reset(max(1, self._estimate_total()))
        append_log(
            self.window.textALLog,
            "오토 라벨 시작 (ultra): "
            f"imgsz=({imgsz_w}x{imgsz_h}), conf={conf}, iou={iou}, approx-eps={approx}, min-area={min_area}, "
            f"viz={viz}, copy_img={copy_img}({copy_mode}), device=auto  → 예상 {self.progress.total}장",
        )
        self._toggle_ui(True)

        config = AutoLabelConfig(
            model_path=self.model_path,
            image_root=self.image_dir,
            save_root=self.output_dir,
            conf=conf,
            iou=iou,
            imgsz_w=imgsz_w,
            imgsz_h=imgsz_h,
            device=None,
            approx_eps=approx,
            min_area=min_area,
            viz=viz,
            copy_images=copy_img,
            copy_mode=copy_mode,
        )
        runner = AutoLabelRunner(config=config, stop_event=self.stop_event, signals=self.signals)
        self.pool.start(runner)

    def _stop(self) -> None:
        if not self.window.btnALStop.isEnabled():
            return
        self.stop_event.set()
        append_log(self.window.textALLog, "오토 라벨 중지 요청을 보냈습니다... (진행 중인 항목은 마무리 후 종료)")

    def _on_one_done(self, ok: bool, message: str) -> None:
        self.progress.update(ok)
        append_log(self.window.textALLog, message)
        self.window.progressAL.setValue(self.progress.percent())

    def _on_all_done(self) -> None:
        self._toggle_ui(False)
        self.window.progressAL.setValue(100)
        append_log(
            self.window.textALLog,
            f"오토 라벨 종료. 처리 {self.progress.done}/{self.progress.total}",
        )
        if self.stop_event.is_set():
            QMessageBox.information(self.window, "오토 라벨 중지됨", "사용자 요청으로 중지되었습니다.")
        else:
            QMessageBox.information(self.window, "오토 라벨 완료", "라벨 생성이 완료되었습니다.")