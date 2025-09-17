# convert.py
# GUI 내에서 LabelMe/AnyLabeling 또는 COCO → YOLOv8 Segmentation 변환 (1280x720 기본, 사각비 지원)
import os, json, random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from PIL import Image
from PyQt5.QtCore import QRunnable
from .signals import Signals

random.seed(0)

# ---------- 레터박스(사각비 지원, 기본 1280x720) ----------
def letterbox(
    im: Image.Image,
    new_shape: Union[int, Tuple[int, int]] = (1280, 720),
    color=(114, 114, 114)
):
    """
    비율 유지 리사이즈 + 패딩 → new_shape로 캔버스 생성.
    - new_shape: int이면 정사각 (S,S), (W,H) 튜플이면 사각.
    반환: (canvas, scale, pad_w, pad_h, (orig_w, orig_h))
    """
    if isinstance(new_shape, int):
        new_w = new_h = int(new_shape)
    else:
        new_w, new_h = map(int, new_shape)

    w, h = im.size
    # 사각 대응: 두 축 중 더 작은 스케일 사용
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    im_resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), color)
    pad_w = (new_w - nw) // 2
    pad_h = (new_h - nh) // 2
    canvas.paste(im_resized, (pad_w, pad_h))

    return canvas, scale, pad_w, pad_h, (w, h)

# 정규화: 출력 폭/높이로 각각 나눔 (사각 해상도 대응)
def norm_xy_rect(x: float, y: float, out_w: int, out_h: int):
    return x / float(out_w), y / float(out_h)

def rect_to_polygon(pts):
    # LabelMe rectangle: [[x1,y1],[x2,y2]]
    (x1, y1), (x2, y2) = pts
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def save_pair(split, out_root: Path, stem, img: Image.Image, lines: List[str]):
    img_out = out_root / f"images/{split}/{stem}.jpg"
    lab_out = out_root / f"labels/{split}/{stem}.txt"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(img_out), quality=95)
    lab_out.write_text("\n".join(lines), encoding="utf-8")

def poly_to_yolo_line(
    cls_id: int,
    poly_xy: List[Tuple[float, float]],
    out_w: int, out_h: int,
    scale: float, pad_w: int, pad_h: int
):
    coords = []
    for x, y in poly_xy:
        xx = x * scale + pad_w
        yy = y * scale + pad_h
        nx, ny = norm_xy_rect(xx, yy, out_w, out_h)
        coords += [f"{nx:.6f}", f"{ny:.6f}"]
    return f"{cls_id} " + " ".join(coords)

# ---------- COCO 탐지 ----------
def detect_coco(jsondir: Path) -> Optional[Dict]:
    """jsondir 안에서 COCO 형식(키: images, annotations, categories) json을 하나 찾아 반환."""
    for p in jsondir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if all(k in obj for k in ("images", "annotations", "categories")):
                return obj
        except Exception:
            pass
    return None

# ---------- 변환 본체들 ----------
def process_coco(
    imgdir: Path, jsondir: Path, out_root: Path,
    new_shape: Union[int, Tuple[int, int]] = (1280, 720),
    val_ratio: float = 0.2,
    signals: Signals = None, stop_flag: callable = lambda: False, total_counter: dict = None
):
    coco = detect_coco(jsondir)
    if coco is None:
        return [], 0

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id2idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    from collections import defaultdict
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    n_ok = n_fail = 0
    for imginfo in coco["images"]:
        if stop_flag():
            break

        file_name = os.path.basename(imginfo.get("file_name", ""))
        src_img = imgdir / file_name
        if not src_img.exists():
            alt = Path(imginfo.get("file_name", ""))
            if alt.exists():
                src_img = alt
            else:
                n_fail += 1
                if signals: signals.one_done.emit(False, f"[MISS] {file_name}")
                if total_counter is not None: total_counter["done"] += 1
                continue

        try:
            im = Image.open(src_img).convert("RGB")
        except Exception as e:
            n_fail += 1
            if signals: signals.one_done.emit(False, f"[FAIL] open {file_name} - {e}")
            if total_counter is not None: total_counter["done"] += 1
            continue

        im_new, scale, pad_w, pad_h, _ = letterbox(im, new_shape)
        out_w, out_h = im_new.size

        lines = []
        for a in ann_by_img.get(imginfo["id"], []):
            cls = cat_id2idx[a["category_id"]]
            seg = a.get("segmentation")

            polys = []
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                for flat in seg:
                    pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
                    if len(pts) >= 3:
                        polys.append(pts)
            elif isinstance(seg, dict):
                x, y, w, h = a["bbox"]
                polys = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
            else:
                x, y, w, h = a["bbox"]
                polys = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]

            for poly in polys:
                if len(poly) >= 3:
                    line = poly_to_yolo_line(cls, poly, out_w, out_h, scale, pad_w, pad_h)
                    lines.append(line)

        split = "val" if random.random() < val_ratio else "train"
        save_pair(split, out_root, Path(file_name).stem, im_new, lines)

        n_ok += 1
        if total_counter is not None: total_counter["done"] += 1
        if signals: signals.one_done.emit(True, f"[OK] {file_name}")

    (out_root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names}\n",
        encoding="utf-8"
    )
    return names, n_ok

def process_labelme(
    imgdir: Path, jsondir: Path, out_root: Path,
    new_shape: Union[int, Tuple[int, int]] = (1280, 720),
    val_ratio: float = 0.2,
    signals: Signals = None, stop_flag: callable = lambda: False, total_counter: dict = None
):
    json_files = list(jsondir.glob("*.json"))

    classes = set()
    metas = []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            for s in data.get("shapes", []):
                classes.add(s["label"])
            metas.append((jp, data))
        except Exception:
            pass
    names = sorted(list(classes))
    name2id = {n: i for i, n in enumerate(names)}

    n_ok = n_fail = 0
    for jp, data in metas:
        if stop_flag():
            break

        img_name = data.get("imagePath") or Path(jp).with_suffix(".jpg").name
        stem = Path(img_name).stem

        # 이미지 찾기
        src_img = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"]:
            cand = imgdir / f"{stem}{ext}"
            if cand.exists():
                src_img = cand
                break
        if not src_img:
            n_fail += 1
            if total_counter is not None: total_counter["done"] += 1
            if signals: signals.one_done.emit(False, f"[MISS] {stem}.*")
            continue

        try:
            im = Image.open(src_img).convert("RGB")
        except Exception as e:
            n_fail += 1
            if total_counter is not None: total_counter["done"] += 1
            if signals: signals.one_done.emit(False, f"[FAIL] open {src_img.name} - {e}")
            continue

        im_new, scale, pad_w, pad_h, _ = letterbox(im, new_shape)
        out_w, out_h = im_new.size

        lines = []
        for s in data.get("shapes", []):
            label = s["label"]
            pts = s.get("points", [])
            st = s.get("shape_type", "polygon")
            if st == "rectangle" and len(pts) == 2:
                poly = rect_to_polygon(pts)
            else:
                poly = [(p[0], p[1]) for p in pts]
            if len(poly) < 3:
                continue
            cls = name2id[label]
            line = poly_to_yolo_line(cls, poly, out_w, out_h, scale, pad_w, pad_h)
            lines.append(line)

        split = "val" if random.random() < val_ratio else "train"
        save_pair(split, out_root, stem, im_new, lines)

        n_ok += 1
        if total_counter is not None: total_counter["done"] += 1
        if signals: signals.one_done.emit(True, f"[OK] {stem}")

    (out_root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names}\n",
        encoding="utf-8"
    )
    return names, n_ok

# ---------- 러너 ----------
class ConvertRunner(QRunnable):
    """GUI에서 클릭 한 번으로 전체 변환 수행. new_shape는 int(정사각) 또는 (W,H) 튜플."""
    def __init__(self, imgdir: Path, jsondir: Path, out_root: Path,
                 size: Union[int, Tuple[int, int]] = (1280, 720), val_ratio: float = 0.2,
                 signals: Signals = None, stop_event=None):
        super().__init__()
        self.imgdir = imgdir
        self.jsondir = jsondir
        self.out_root = out_root
        # size=int 또는 (w,h) 지원. 기본 (1280,720)
        self.new_shape = size if isinstance(size, (tuple, list)) else int(size)
        self.val_ratio = float(val_ratio)
        self.signals = signals
        self.stop_event = stop_event

    def stop_flag(self):
        return self.stop_event.is_set() if self.stop_event else False

    def run(self):
        try:
            self.out_root.mkdir(parents=True, exist_ok=True)
            for p in ["images/train", "images/val", "labels/train", "labels/val"]:
                (self.out_root / p).mkdir(parents=True, exist_ok=True)

            total_counter = {"done": 0}
            coco_obj = detect_coco(self.jsondir)
            if coco_obj:
                process_coco(
                    imgdir=self.imgdir, jsondir=self.jsondir,
                    out_root=self.out_root, new_shape=self.new_shape,
                    val_ratio=self.val_ratio,
                    signals=self.signals, stop_flag=self.stop_flag,
                    total_counter=total_counter,
                )
            else:
                process_labelme(
                    imgdir=self.imgdir, jsondir=self.jsondir,
                    out_root=self.out_root, new_shape=self.new_shape,
                    val_ratio=self.val_ratio,
                    signals=self.signals, stop_flag=self.stop_flag,
                    total_counter=total_counter,
                )

        except Exception as e:
            if self.signals: self.signals.one_done.emit(False, f"[FAIL] convert - {e}")
        finally:
            if self.signals: self.signals.all_done.emit()
