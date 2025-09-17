"""Controller for the label verification tab."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QShortcut

from ...core.files import copy_file, list_image_files
from .common import append_log


@dataclass
class ReviewItem:
    """Represents an image/label pair being reviewed."""

    image: Path
    label: Path
    relative: Path
    is_good: bool = False
    warned: bool = False


class LabelReviewTabController:
    """UI controller that lets users inspect and sort YOLO-Seg labels."""

    COLOUR_TABLE: Sequence[tuple[int, int, int]] = (
        (0, 255, 0),
        (255, 0, 0),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
    )

    def __init__(self, window, thread_pool) -> None:
        self.window = window
        self.pool = thread_pool

        self.image_dir: Path | None = None
        self.label_dir: Path | None = None
        self.good_dir: Path | None = None
        self.bad_dir: Path | None = None

        self.items: List[ReviewItem] = []
        self.current_index: int = -1

        self.window.labelReviewPaths.setAlignment(Qt.AlignLeft)
        self._update_paths_label()
        self.window.labelReviewStatus.setText("0/0 · 정상 표시 0 · 미분류 0")
        self.window.labelReviewInfo.setText("검수할 파일을 불러오세요.")
        self.window.labelReviewImage.setAlignment(Qt.AlignCenter)
        self.window.labelReviewImage.setText("이미지가 여기에 표시됩니다.")
        self.window.textReviewLog.setReadOnly(True)
        self.window.textReviewLog.setPlaceholderText("검수 로그가 표시됩니다...")

        self.window.btnReviewImages.clicked.connect(self._select_images)
        self.window.btnReviewLabels.clicked.connect(self._select_labels)
        self.window.btnReviewGood.clicked.connect(self._select_good_dir)
        self.window.btnReviewBad.clicked.connect(self._select_bad_dir)
        self.window.btnReviewPrev.clicked.connect(self._go_prev)
        self.window.btnReviewNext.clicked.connect(self._go_next)
        self.window.btnReviewMarkGood.clicked.connect(self._mark_good)
        self.window.btnReviewClearGood.clicked.connect(self._clear_good)
        self.window.btnReviewFinalize.clicked.connect(self._finalize)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self.window.tabReview)
        self.shortcut_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_prev.activated.connect(self._go_prev)
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self.window.tabReview)
        self.shortcut_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_next.activated.connect(self._go_next)
        self.shortcut_toggle = QShortcut(QKeySequence(Qt.Key_Space), self.window.tabReview)
        self.shortcut_toggle.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_toggle.activated.connect(self._toggle_good_shortcut)

        self._update_controls()

    # UI helpers --------------------------------------------------------------
    def _log(self, message: str) -> None:
        append_log(self.window.textReviewLog, message)

    def _current_item(self) -> ReviewItem | None:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None

    def _update_paths_label(self) -> None:
        def fmt(path: Path | None) -> str:
            return str(path) if path else "(미선택)"

        lines = [
            f"이미지: {fmt(self.image_dir)}",
            f"라벨: {fmt(self.label_dir)}",
            f"정상 폴더: {fmt(self.good_dir)}",
            f"이상 폴더: {fmt(self.bad_dir)}",
        ]
        self.window.labelReviewPaths.setText("\n".join(lines))

    def _update_status_labels(self) -> None:
        total = len(self.items)
        good = sum(1 for item in self.items if item.is_good)
        remaining = total - good
        if total > 0 and 0 <= self.current_index < total:
            position = f"{self.current_index + 1}/{total}"
        else:
            position = "0/0"

        self.window.labelReviewStatus.setText(
            f"{position} · 정상 표시 {good} · 미분류 {remaining}"
        )

        current = self._current_item()
        if current:
            state = "정상" if current.is_good else "미분류"
            self.window.labelReviewInfo.setText(
                f"파일: {current.relative.as_posix()} · 상태: {state}"
            )
        else:
            self.window.labelReviewInfo.setText("검수할 파일을 불러오세요.")

    def _update_controls(self) -> None:
        has_items = bool(self.items)
        self.window.btnReviewPrev.setEnabled(has_items and self.current_index > 0)
        self.window.btnReviewNext.setEnabled(has_items and self.current_index < len(self.items) - 1)

        current = self._current_item()
        self.window.btnReviewMarkGood.setEnabled(bool(current and not current.is_good))
        self.window.btnReviewClearGood.setEnabled(bool(current and current.is_good))

        can_finalize = has_items and self.good_dir and self.bad_dir
        self.window.btnReviewFinalize.setEnabled(bool(can_finalize))

    def _clear_items(self) -> None:
        self.items = []
        self.current_index = -1
        self.window.labelReviewImage.clear()
        self.window.labelReviewImage.setText("이미지가 여기에 표시됩니다.")
        self._update_status_labels()
        self._update_controls()

    # Directory selection -----------------------------------------------------
    def _select_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "이미지 폴더 선택")
        if not directory:
            return
        self.image_dir = Path(directory)
        self._log(f"[DIR] 이미지 폴더: {self.image_dir}")
        self._update_paths_label()
        if self.label_dir:
            self._load_items()
        else:
            self._clear_items()

    def _select_labels(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "라벨 폴더 선택")
        if not directory:
            return
        self.label_dir = Path(directory)
        self._log(f"[DIR] 라벨 폴더: {self.label_dir}")
        self._update_paths_label()
        if self.image_dir:
            self._load_items()
        else:
            self._clear_items()

    def _select_good_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "정상 폴더 선택")
        if not directory:
            return
        self.good_dir = Path(directory)
        self._log(f"[DIR] 정상 폴더: {self.good_dir}")
        self._update_paths_label()
        self._update_controls()

    def _select_bad_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "이상 폴더 선택")
        if not directory:
            return
        self.bad_dir = Path(directory)
        self._log(f"[DIR] 이상 폴더: {self.bad_dir}")
        self._update_paths_label()
        self._update_controls()

    # Dataset handling --------------------------------------------------------
    def _load_items(self) -> None:
        if not (self.image_dir and self.label_dir):
            self._clear_items()
            return

        image_files = list_image_files(self.image_dir)
        items: List[ReviewItem] = []
        missing: List[str] = []

        for image_path in image_files:
            try:
                relative = image_path.relative_to(self.image_dir)
            except ValueError:
                relative = Path(image_path.name)

            label_path = self.label_dir / relative.with_suffix(".txt")
            if label_path.exists():
                items.append(ReviewItem(image=image_path, label=label_path, relative=relative))
            else:
                missing.append(relative.as_posix())

        self.items = items
        self.current_index = 0 if self.items else -1

        if missing:
            sample = ", ".join(missing[:3])
            remainder = len(missing) - 3
            if remainder > 0:
                sample = f"{sample} 외 {remainder}건"
            self._log(f"[WARN] 라벨 없음: {sample}")

        if self.items:
            self._log(f"[INFO] 총 {len(self.items)}건을 불러왔습니다.")
        else:
            self._log("[INFO] 검수할 이미지가 없습니다.")

        self._show_current_image()

    # Navigation --------------------------------------------------------------
    def _go_prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()

    def _go_next(self) -> None:
        if self.current_index < len(self.items) - 1:
            self.current_index += 1
            self._show_current_image()

    def _toggle_good_shortcut(self) -> None:
        current = self._current_item()
        if not current:
            return
        if current.is_good:
            self._clear_good()
        else:
            self._mark_good()

    # Marking -----------------------------------------------------------------
    def _mark_good(self) -> None:
        current = self._current_item()
        if not current or current.is_good:
            return
        current.is_good = True
        self._log(f"[GOOD] {current.relative.as_posix()}")
        if self.current_index < len(self.items) - 1:
            self.current_index += 1
        self._show_current_image()

    def _clear_good(self) -> None:
        current = self._current_item()
        if not current or not current.is_good:
            return
        current.is_good = False
        self._log(f"[UNDO] {current.relative.as_posix()}")
        self._show_current_image()

    # Visualisation -----------------------------------------------------------
    def _show_current_image(self) -> None:
        current = self._current_item()
        if not current:
            self.window.labelReviewImage.clear()
            self.window.labelReviewImage.setText("이미지가 여기에 표시됩니다.")
            self._update_status_labels()
            self._update_controls()
            return

        pixmap = self._build_pixmap(current)
        if pixmap:
            self.window.labelReviewImage.setText("")
            self.window.labelReviewImage.setPixmap(pixmap)
            self.window.labelReviewImage.adjustSize()
        else:
            self.window.labelReviewImage.clear()
            self.window.labelReviewImage.setText("이미지를 불러오지 못했습니다.")

        self._update_status_labels()
        self._update_controls()

    def _build_pixmap(self, item: ReviewItem) -> QPixmap | None:
        try:
            with Image.open(item.image) as im:
                rgb_image = im.convert("RGB")
                width, height = rgb_image.size
                image_bgr = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._log(f"[ERR] 이미지 로드 실패: {item.relative.as_posix()} - {exc}")
            return None

        polygons, classes, warnings = self._parse_label(item, width, height)
        if warnings and not item.warned:
            for warn in warnings:
                self._log(f"[WARN] {item.relative.as_posix()} - {warn}")
            item.warned = True

        canvas = image_bgr.copy()
        for pts, cls_id in zip(polygons, classes):
            pts_int = np.round(pts).astype(np.int32)
            if pts_int.shape[0] < 3:
                continue
            colour = self._colour_for_class(cls_id)
            cv2.polylines(canvas, [pts_int], True, colour, 2)
            x, y = pts_int[0]
            label = str(cls_id)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            box_x = int(max(0, min(x, canvas.shape[1] - text_w - 6)))
            box_y = int(max(text_h + 4, min(y, canvas.shape[0] - 2)))
            cv2.rectangle(
                canvas,
                (box_x, box_y - text_h - 4),
                (box_x + text_w + 4, box_y),
                colour,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (box_x + 2, box_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        bytes_per_line = int(rgb.strides[0])
        qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    def _parse_label(
        self, item: ReviewItem, width: int, height: int
    ) -> tuple[List[np.ndarray], List[int], List[str]]:
        polygons: List[np.ndarray] = []
        classes: List[int] = []
        warnings: List[str] = []

        try:
            try:
                text = item.label.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = item.label.read_text(encoding="utf-8-sig")
        except Exception as exc:  # pragma: no cover - UI feedback
            warnings.append(f"라벨 파일을 읽을 수 없습니다: {exc}")
            return polygons, classes, warnings

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                warnings.append(f"{line_no}행: 좌표 개수가 부족합니다.")
                continue

            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                warnings.append(f"{line_no}행: 클래스 ID를 해석할 수 없습니다.")
                continue

            coord_values = parts[1:]
            if len(coord_values) % 2 != 0:
                warnings.append(f"{line_no}행: 좌표 값의 개수가 짝수가 아닙니다.")
                continue

            try:
                coords = [float(v) for v in coord_values]
            except ValueError:
                warnings.append(f"{line_no}행: 좌표 값을 해석할 수 없습니다.")
                continue

            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                warnings.append(f"{line_no}행: 유효한 폴리곤이 아닙니다.")
                continue

            pts[:, 0] = np.clip(pts[:, 0] * width, 0, max(width - 1, 0))
            pts[:, 1] = np.clip(pts[:, 1] * height, 0, max(height - 1, 0))

            polygons.append(pts)
            classes.append(cls_id)

        if not polygons:
            warnings.append("유효한 폴리곤이 없습니다.")

        return polygons, classes, warnings

    def _colour_for_class(self, cls_id: int) -> tuple[int, int, int]:
        table = self.COLOUR_TABLE
        return table[cls_id % len(table)]

    # Export ------------------------------------------------------------------
    def _finalize(self) -> None:
        if not self.items:
            QMessageBox.information(self.window, "알림", "검수할 항목이 없습니다.")
            return
        if not self.good_dir or not self.bad_dir:
            QMessageBox.warning(self.window, "경고", "정상/이상 폴더를 모두 선택하세요.")
            return
        if self.good_dir == self.bad_dir:
            QMessageBox.warning(self.window, "경고", "정상 폴더와 이상 폴더는 서로 달라야 합니다.")
            return

        good_count = sum(1 for item in self.items if item.is_good)
        bad_count = len(self.items) - good_count

        reply = QMessageBox.question(
            self.window,
            "폴더 분리",
            (
                f"정상 {good_count}건, 이상 {bad_count}건을 각각의 폴더로 복사합니다.\n"
                "계속할까요?"
            ),
        )
        if reply != QMessageBox.Yes:
            return

        errors = 0
        for item in self.items:
            dest_root = self.good_dir if item.is_good else self.bad_dir
            image_out = dest_root / "images" / item.relative
            label_out = dest_root / "labels" / item.relative.with_suffix(".txt")
            try:
                copy_file(item.image, image_out)
                copy_file(item.label, label_out)
            except Exception as exc:  # pragma: no cover - UI feedback
                errors += 1
                self._log(f"[ERR] 복사 실패: {item.relative.as_posix()} - {exc}")

        self._log(f"[DONE] 정상 {good_count}건 → {self.good_dir}")
        self._log(f"[DONE] 이상 {bad_count}건 → {self.bad_dir}")

        if errors:
            QMessageBox.warning(
                self.window,
                "분리 완료 (일부 실패)",
                f"복사 중 {errors}건의 오류가 발생했습니다. 로그를 확인하세요.",
            )
        else:
            QMessageBox.information(
                self.window,
                "분리 완료",
                f"정상 {good_count}건 / 이상 {bad_count}건으로 분리했습니다.",
            )