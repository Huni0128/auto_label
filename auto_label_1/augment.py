import os
import json
import base64
import threading
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from PyQt5.QtCore import QRunnable

from .signals import Signals

# ===== 출력 해상도(검증 영상과 일치) =====
TARGET_W = 1280
TARGET_H = 720

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ===== Albumentations 파이프라인 =====
# - 최종 해상도를 1280x720으로 고정 (검증 영상과 동일)
def build_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.0),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.2, rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            # 1280x720 고정 (가로 1280, 세로 720). 필요 시 PadIfNeeded로 비율 유지+패딩 방식으로 변경 가능.
            A.Resize(TARGET_H, TARGET_W, interpolation=cv2.INTER_AREA, p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

# (참고) 비율 유지 + 패딩으로 1280x720 맞추려면 아래처럼 교체:
# A.LongestMaxSize(max_size=TARGET_W, interpolation=cv2.INTER_AREA, p=1.0),
# A.PadIfNeeded(min_height=TARGET_H, min_width=TARGET_W, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),


# ===== 유틸 =====
def imread_any(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)

def imdecode_from_labelme_imagedata(imageData: str):
    data = base64.b64decode(imageData)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def clip_point(pt, w, h):
    x, y = pt
    return (max(0, min(float(x), w - 1)), max(0, min(float(y), h - 1)))

def sanitize_polygon(points, w, h, min_area=5.0):
    clean = []
    for (x, y) in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx, cy = clip_point((x, y), w, h)
        if len(clean) == 0 or (abs(cx - clean[-1][0]) > 1e-3 or abs(cy - clean[-1][1]) > 1e-3):
            clean.append((cx, cy))
    if len(clean) < 3:
        return []
    if polygon_area(clean) < float(min_area):
        return []
    return clean

def guess_image_for_json(json_path: Path, in_dir: Path):
    """
    1) JSON의 imagePath를 우선 신뢰
    2) imageData가 있으면 파일 없이 진행
    3) 동일 stem 파일 탐색
    """
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # 1) imagePath 기반
    if "imagePath" in j and isinstance(j["imagePath"], str) and j["imagePath"]:
        p = (in_dir / j["imagePath"]).resolve()
        if p.exists():
            return p, j

    # 2) imageData 기반
    if "imageData" in j and j["imageData"]:
        return None, j

    # 3) 같은 stem 탐색
    stem = json_path.stem
    for ext in IMG_EXTS:
        cand = (in_dir / f"{stem}{ext}")
        if cand.exists():
            return cand, j

    return None, j


# ===== GUI용 작업 스레드 =====
class AugmentTask(QRunnable):
    """
    한 JSON 파일을 받아 multiplier 만큼 증강 샘플을 생성
    - 진행/로그는 Signals.one_done으로 샘플마다 1회 emit
    """
    def __init__(
        self,
        json_path: Path,
        in_dir: Path,
        out_dir: Path,
        multiplier: int,
        stop_event: threading.Event,
        signals: Signals,
    ):
        super().__init__()
        self.json_path = json_path
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.multiplier = int(max(1, multiplier))
        self.stop_event = stop_event
        self.signals = signals

    def run(self):
        if self.stop_event.is_set():
            self.signals.one_done.emit(False, f"[STOPPED] {self.json_path.name}")
            return

        try:
            img_path, j = guess_image_for_json(self.json_path, self.in_dir)

            # 이미지 로드
            image = None
            if img_path is not None:
                image = imread_any(img_path)
            if image is None and j.get("imageData"):
                image = imdecode_from_labelme_imagedata(j["imageData"])
            if image is None:
                self.signals.one_done.emit(False, f"[FAIL] 이미지 로드 실패: {self.json_path.name}")
                return

            # polygon 수집
            shapes = j.get("shapes", [])
            poly_indices = []
            polygons = []
            for idx, shp in enumerate(shapes):
                stype = shp.get("shape_type", "polygon")
                if stype != "polygon":
                    continue
                pts = shp.get("points", [])
                if not pts or len(pts) < 3:
                    continue
                pts_xy = [(float(x), float(y)) for x, y in pts]
                polygons.append(pts_xy)
                poly_indices.append(idx)

            # keypoints로 평탄화
            flat_kps = [pt for poly in polygons for pt in poly]
            layout = []
            cursor = 0
            for poly in polygons:
                start = cursor
                cursor += len(poly)
                end = cursor
                layout.append((start, end))  # flat_kps[start:end] = 한 폴리곤

            base_stem = img_path.stem if img_path is not None else self.json_path.stem
            orig_ext = img_path.suffix if img_path is not None else ".png"

            # 변환기 (태스크별 인스턴스)
            transforms = build_transforms()

            for k in range(self.multiplier):
                if self.stop_event.is_set():
                    self.signals.one_done.emit(False, f"[STOPPED] {self.json_path.stem} (k={k+1})")
                    break

                kps_input = flat_kps if len(flat_kps) > 0 else [(0.0, 0.0)]
                transformed = transforms(image=image, keypoints=kps_input)
                img_aug = transformed["image"]
                kps_aug = transformed["keypoints"]
                H, W = img_aug.shape[:2]

                # 폴리곤 복원
                new_shapes = deepcopy(shapes)
                if len(flat_kps) > 0 and len(kps_aug) >= 1:
                    kps_np = np.array(kps_aug, dtype=np.float32)
                    for li, (s, e) in enumerate(layout):
                        pts_aug = [(float(kps_np[i, 0]), float(kps_np[i, 1])) for i in range(s, e)]
                        pts_aug = sanitize_polygon(pts_aug, W, H, min_area=5.0)
                        target_idx = poly_indices[li]
                        if len(pts_aug) >= 3:
                            new_shapes[target_idx]["points"] = [[float(x), float(y)] for x, y in pts_aug]
                            new_shapes[target_idx]["shape_type"] = "polygon"
                        else:
                            # 깨진 폴리곤은 제거
                            new_shapes[target_idx]["points"] = []
                    # 빈 폴리곤 삭제
                    new_shapes = [s for s in new_shapes if not (s.get("shape_type", "polygon") == "polygon" and len(s.get("points", [])) < 3)]
                else:
                    # 폴리곤 없음 → 폴리곤 타입 제거(좌표 불일치 방지)
                    new_shapes = [s for s in new_shapes if s.get("shape_type", "polygon") != "polygon"]

                # 출력 파일명
                out_img_name = f"{base_stem}_aug{k+1}{orig_ext}"
                out_json_name = f"{base_stem}_aug{k+1}.json"

                # 저장
                cv2.imwrite(str(self.out_dir / out_img_name), img_aug)

                new_j = deepcopy(j)
                new_j["imagePath"] = out_img_name
                new_j["imageWidth"] = int(W)   # = 1280
                new_j["imageHeight"] = int(H)  # = 720
                new_j["imageData"] = None  # 용량 절감
                new_j["shapes"] = new_shapes

                with open(self.out_dir / out_json_name, "w", encoding="utf-8") as f:
                    json.dump(new_j, f, ensure_ascii=False, indent=2)

                self.signals.one_done.emit(True, f"[OK] {out_img_name}")

        except Exception as e:
            self.signals.one_done.emit(False, f"[FAIL] {self.json_path.name} - {e}")
