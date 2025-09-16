# auto_labeler.py
# Ultralytics YOLO Seg 결과(r.masks.xy)를 "ultra" 방식으로만 사용하여
# YOLO-seg .txt 및 (선택) 시각화를 저장합니다.
import os
import sys
import math
import shutil
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QRunnable

from .signals import Signals

try:
    from ultralytics import YOLO
except ImportError:
    print("`pip install ultralytics` 먼저 설치하세요.", file=sys.stderr)
    raise

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ===== score → bin 폴더 규칙 =====
BIN_LO = 0.60
BIN_STEP = 0.05
BIN_TOP = 0.95

def score_to_bucket(score: float) -> str:
    if score < BIN_LO:
        return "lt_0.60"
    if score >= BIN_TOP:
        return "ge_0.95"
    k = int(math.floor((score - BIN_LO) / BIN_STEP))
    start = BIN_LO + k * BIN_STEP
    end = start + BIN_STEP
    return f"{start:.2f}_{end:.2f}"

# ===== 유틸 =====
def list_images(p: Path) -> List[Path]:
    if p.is_dir():
        fs = [x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS]
        fs.sort()
        return fs
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    raise FileNotFoundError(f"이미지/폴더 경로 오류: {p}")

def clip_points(points, W, H):
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts

def to_labelme_shapes(polys, label, class_id, score, min_area):
    shapes = []
    for poly in polys:
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim == 3:
            poly = poly.reshape(-1, 2)
        if len(poly) < 3:
            continue
        area = float(abs(cv2.contourArea(poly)))
        if area < min_area:
            continue
        shapes.append({
            "label": label,
            "points": poly.astype(float).tolist(),
            "group_id": int(class_id),
            "shape_type": "polygon",
            "flags": {"class_id": int(class_id), "score": float(score), "area_px2": area}
        })
    return shapes

def save_yolo_seg_txt(path: Path, W: int, H: int, shapes):
    lines = []
    for s in shapes:
        pts = np.array(s["points"], dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] = pts[:, 0] / float(W)
        pts[:, 1] = pts[:, 1] / float(H)
        pts = np.clip(pts, 0.0, 1.0)
        flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
        class_id = int(s["flags"].get("class_id", s.get("group_id", 0)))
        lines.append(f"{class_id} {flat}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def draw_viz(img_bgr, shapes):
    vis = img_bgr.copy()
    for s in shapes:
        pts = np.array(s["points"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            txt = f"{s['label']} {s['flags'].get('score', 0):.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(th + 4, y)
            cv2.rectangle(vis, (x, y - th - 4), (x + tw + 4, y), (0, 255, 0), -1)
            cv2.putText(vis, txt, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis

def get_ultra_segments(result, i, approx_eps=0.0):
    """
    YOLO r.masks.xy[i]의 폴리곤 좌표를 그대로 사용(ultra).
    필요 시 approx-eps로 RDP 단순화.
    """
    segs = result.masks.xy[i]
    polys = []
    if segs is None:
        return polys
    if isinstance(segs, np.ndarray):
        segs = [segs]
    for seg in segs:
        if seg is None or len(seg) < 3:
            continue
        cnt = np.asarray(seg, dtype=np.float32).reshape(-1, 1, 2)
        if approx_eps > 0:
            cnt = cv2.approxPolyDP(cnt, approx_eps, True)
        polys.append(cnt.reshape(-1, 2))
    return polys


class AutoLabelRunner(QRunnable):
    """
    오토 라벨 작업 러너 (ultra 전용)
    - Ultralytics YOLO Seg의 r.masks.xy만 사용
    - mask-thr / morph / retree / contour 경로 제거
    """
    def __init__(
        self,
        model_path: Path,
        img_root: Path,
        save_root: Path,
        conf: float,
        iou: float,
        imgsz_w: int,
        imgsz_h: int,
        device: str,               # '' 또는 None 이면 자동
        approx_eps: float,
        min_area: float,
        viz: bool,
        copy_img: bool,
        copy_mode: str,            # 'copy' | 'hardlink' | 'symlink' | 'move'
        signals: Signals,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.img_root = Path(img_root)
        self.save_root = Path(save_root)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_w = int(imgsz_w)
        self.imgsz_h = int(imgsz_h)
        self.device = device if device else None
        self.approx_eps = float(approx_eps)
        self.min_area = float(min_area)
        self.viz = bool(viz)
        self.copy_img = bool(copy_img)
        self.copy_mode = copy_mode
        self.signals = signals
        self.stop_event = stop_event

    def _emit(self, ok: bool, msg: str):
        if self.signals:
            self.signals.one_done.emit(ok, msg)

    def _copy_image(self, src: Path, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.copy_mode == "copy":
                shutil.copy2(str(src), str(dest))
            elif self.copy_mode == "hardlink":
                os.link(str(src), str(dest))
            elif self.copy_mode == "symlink":
                try:
                    os.symlink(os.path.abspath(str(src)), str(dest))
                except OSError:
                    shutil.copy2(str(src), str(dest))
            elif self.copy_mode == "move":
                shutil.move(str(src), str(dest))
        except Exception as e:
            self._emit(False, f"[{src.name}] 이미지 보관 실패: {e}")

    def run(self):
        try:
            imgs = list_images(self.img_root)
        except Exception as e:
            self._emit(False, f"[ERR] 입력 이미지 검색 실패: {e}")
            if self.signals:
                self.signals.all_done.emit()
            return

        try:
            model = YOLO(str(self.model_path))
            names = getattr(model, "names", None)
            class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else (names or [])
        except Exception as e:
            self._emit(False, f"[ERR] 모델 로드 실패: {e}")
            if self.signals:
                self.signals.all_done.emit()
            return

        total = len(imgs)
        if total == 0:
            self._emit(False, "[INFO] 처리할 이미지가 없습니다.")
            if self.signals:
                self.signals.all_done.emit()
            return

        for idx, img_path in enumerate(imgs, 1):
            if self.stop_event.is_set():
                self._emit(False, "[STOP] 사용자 중지 요청")
                break
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception as e:
                self._emit(False, f"[{img_path.name}] 이미지 열기 실패: {e}")
                continue

            # imgsz: 직사각이면 (H,W)
            imgsz = (self.imgsz_h, self.imgsz_w) if (self.imgsz_w != self.imgsz_h) else int(self.imgsz_w)

            try:
                res = model.predict(
                    source=str(img_path),
                    imgsz=imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e:
                self._emit(False, f"[{img_path.name}] 예측 실패: {e}")
                continue

            if not res:
                # 결과 없음 → 빈 txt
                bucket = "lt_0.60"
                out_txt_dir = self.save_root / bucket / "yolo-seg"
                out_viz_dir = self.save_root / bucket / "viz"
                tp = out_txt_dir / f"{img_path.stem}.txt"
                out_txt_dir.mkdir(parents=True, exist_ok=True)
                tp.write_text("", encoding="utf-8")

                if self.copy_img:
                    self._copy_image(img_path, out_txt_dir / img_path.name)
                if self.viz:
                    img = cv2.imread(str(img_path))
                    out_viz_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_viz_dir / img_path.name), img)

                self._emit(True, f"[{img_path.name}] no result -> bin={bucket} -> SAVE empty .txt  ({idx}/{total})")
                continue

            r = res[0]
            shapes = []

            # ★ ultra 전용: r.masks.xy만 사용
            if getattr(r, "masks", None) is not None and r.masks is not None and r.masks.data is not None and r.masks.xy is not None:
                N = r.masks.data.shape[0]
                cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((N,), int)
                confs = r.boxes.conf.cpu().numpy() if (r.boxes is not None and r.boxes.conf is not None) else np.ones((N,), float)

                for i in range(N):
                    c = int(cls_ids[i]) if i < len(cls_ids) else 0
                    score = float(confs[i]) if i < len(confs) else 1.0
                    label = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"

                    polys = get_ultra_segments(r, i, approx_eps=self.approx_eps)
                    if polys:
                        clipped = [clip_points(p, W, H) for p in polys]
                        shapes += to_labelme_shapes(clipped, label, c, score, self.min_area)

            # 저장/로그
            if len(shapes) > 0:
                scores = [float(s.get("flags", {}).get("score", 1.0)) for s in shapes]
                agg = min(scores) if scores else 1.0
                bucket = score_to_bucket(agg)

                out_txt_dir = self.save_root / bucket / "yolo-seg"
                out_viz_dir = self.save_root / bucket / "viz"

                tp = out_txt_dir / f"{img_path.stem}.txt"
                save_yolo_seg_txt(tp, W, H, shapes)

                if self.copy_img:
                    self._copy_image(img_path, out_txt_dir / img_path.name)
                if self.viz:
                    img = cv2.imread(str(img_path))
                    vis = draw_viz(img, shapes)
                    out_viz_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_viz_dir / img_path.name), vis)

                self._emit(True, f"[{img_path.name}] segments={len(shapes)} score(min)={agg:.3f} -> bin={bucket} -> SAVE .txt  ({idx}/{total})")
            else:
                bucket = "lt_0.60"
                out_txt_dir = self.save_root / bucket / "yolo-seg"
                out_viz_dir = self.save_root / bucket / "viz"
                tp = out_txt_dir / f"{img_path.stem}.txt"
                out_txt_dir.mkdir(parents=True, exist_ok=True)
                tp.write_text("", encoding="utf-8")

                if self.copy_img:
                    self._copy_image(img_path, out_txt_dir / img_path.name)
                if self.viz:
                    img = cv2.imread(str(img_path))
                    out_viz_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_viz_dir / img_path.name), img)

                self._emit(True, f"[{img_path.name}] segments=0 -> bin={bucket} -> SAVE empty .txt  ({idx}/{total})")

        if self.signals:
            self.signals.all_done.emit()
