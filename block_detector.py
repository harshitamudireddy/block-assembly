"""
Block Detector module: Localizes and identifies individual blocks.
Supports:
  1. Deep learning YOLOv8 object detection (when weights are available in runs_detect/block_detector/weights/best.pt)
  2. Plastic-ratio color segmentation engine (zero-training fallback, skin-tone invariant)
  3. Incoming part tracking (isolates object in hand from the assembled cluster)
"""

import os
import cv2
import numpy as np
from block_config import BLOCK_CLASSES, BLOCK_COLORS_BGR, PROJECT_ROOT

YOLO_DET_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs_detect", "block_detector", "weights", "best.pt")


class BlockDetector:
    def __init__(self, model_path=YOLO_DET_MODEL_PATH, conf_threshold=0.40):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.yolo_model = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                print(f"[BlockDetector] Loaded trained YOLO model from '{model_path}'")
            except Exception as e:
                print(f"[BlockDetector] Could not load YOLO model ({e}). Using plastic-ratio detector.")
        else:
            print("[BlockDetector] No trained YOLO weights found. Using plastic-ratio color detector.")

    def is_real_block(self, detection, img):
        """
        Validates a candidate block detection to reject skin tones, hand noise,
        and tiny spurious bounding boxes.
        """
        if img is None or not detection:
            return False

        cname = detection.get("class_name")
        conf = detection.get("confidence", 1.0)
        bx, by, bw, bh = detection["bbox"]
        h_img, w_img = img.shape[:2]

        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        min_dim = min(x2 - x1, y2 - y1)
        max_dim = max(x2 - x1, y2 - y1)
        if min_dim < 14 or (bw * bh) < 600:
            return False

        # Reject extreme aspect ratios (unless high confidence where block is partially occluded under a beam)
        aspect = max_dim / max(min_dim, 1)
        if aspect > 4.5 or (aspect > 3.6 and conf < 0.65):
            return False

        # Reject HUD bar / edge button artifacts (e.g. at top/bottom border)
        if (by < 30 or by > h_img - 40) and (bw > 250 or aspect > 2.8):
            return False
        if y2 > h_img - 35 and bh < 35:
            return False

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        # Color & Skin-Tone Validation for plastic toy blocks vs human skin / clutter
        b, g, r = cv2.split(crop)
        mr = float(np.mean(r))
        mg = float(np.mean(g))
        mb = float(np.mean(b))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 175) & (cb >= 80) & (cb <= 128)
        skin_ratio = float(np.sum(skin_mask)) / float(crop.shape[0] * crop.shape[1])

        # 1. Yellow Block: High saturation, high Red+Green, low Blue.
        # Strictly rejects human hands, palms, and fingers which have low saturation and high blue.
        if cname == "yellow_block":
            if sat < 80:
                return False
            if mr < 90 or mg < 70:
                return False
            # Yellow plastic has very low blue compared to red and green
            if mr < 1.30 * mb or mg < 1.15 * mb:
                return False
            # Reject if human skin tone is present
            if skin_ratio > 0.40:
                return False

        # 2. Red Block: Strong red dominance and saturation. Reject reddish skin.
        elif cname == "red_block":
            if mr < 118 or mr <= mg or mr <= mb:
                return False
            if sat < 75:
                return False
            if skin_ratio > 0.65:
                return False
            if conf < 0.85:
                if sat < 100 or mr < 1.20 * mg or mr < 1.20 * mb:
                    return False

        # 3. Blue Block: Blue dominant over red
        elif cname == "blue_block":
            if mb < 60:
                return False
            if mb < 1.02 * mr and mb < 1.02 * mg:
                return False
            if sat < 35:
                return False

        # 4. Green Block: Green dominant over red and blue
        elif cname == "green_block":
            if mg < 55:
                return False
            if mg < 1.02 * mr and mg < 1.02 * mb:
                return False
            if sat < 35:
                return False

        return True

    def detect_color_plastic(self, img, min_area=600):
        """
        Segment blocks using high-saturation plastic color ratios that strictly reject human skin.
        Returns: list of dicts {class_name, class_id, bbox: (x, y, w, h), center: (cx, cy), area, confidence, source}
        """
        h, w = img.shape[:2]
        b, g, r = cv2.split(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # 1. Blue: High Blue channel relative to Red and Green, Hue 90-135
        blue_mask = (
            (b > 75)
            & (b > 1.25 * r.astype(np.float32))
            & (b > 1.15 * g.astype(np.float32))
            & (hue >= 90)
            & (hue <= 135)
            & (sat > 65)
        )

        # 2. Green: High Green channel, Hue 35-85
        green_mask = (
            (g > 70)
            & (g > 1.20 * r.astype(np.float32))
            & (g > 1.15 * b.astype(np.float32))
            & (hue >= 35)
            & (hue <= 85)
            & (sat > 60)
        )

        # 3. Red: High Red channel relative to Green & Blue, high saturation (strictly eliminates skin)
        red_mask = (
            (r > 100)
            & (r > 1.65 * g.astype(np.float32))
            & (r > 1.65 * b.astype(np.float32))
            & ((hue <= 12) | (hue >= 168))
            & (sat > 115)
            & (val > 65)
        )

        # 4. Yellow: High Red & Green, low Blue, Hue 16-34
        yellow_mask = (
            (r > 100)
            & (g > 80)
            & (r > 1.45 * b.astype(np.float32))
            & (g > 1.35 * b.astype(np.float32))
            & (hue >= 16)
            & (hue <= 34)
            & (sat > 95)
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        masks = {
            "blue_block": blue_mask.astype(np.uint8) * 255,
            "green_block": green_mask.astype(np.uint8) * 255,
            "red_block": red_mask.astype(np.uint8) * 255,
            "yellow_block": yellow_mask.astype(np.uint8) * 255,
        }

        detections = []
        for cls_name, raw_mask in masks.items():
            cls_id = BLOCK_CLASSES.index(cls_name)
            # Morphological cleaning
            cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= min_area:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect > 4.5:
                        continue

                    cx, cy = bx + bw // 2, by + bh // 2
                    hull = cv2.convexHull(c)
                    solidity = float(area) / max(cv2.contourArea(hull), 1)
                    conf = min(0.96, max(0.55, 0.45 + 0.50 * solidity))

                    detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "bbox": (bx, by, bw, bh),
                        "center": (cx, cy),
                        "area": area,
                        "confidence": float(conf),
                        "source": "color_plastic",
                    })

        # Sort largest area first
        detections.sort(key=lambda d: d["area"], reverse=True)
        return self.split_merged_blocks(detections, img)

    def split_merged_block(self, detection, img):
        """
        Splits a merged block detection (red_block or blue_block) into two individual detections
        when it physically encompasses two adjacent or stacked blocks.
        Strictly respects orientation:
          - cw > ch: cuts vertically along X into Left and Right blocks (never top & bottom).
          - ch > cw: cuts horizontally along Y into Top and Bottom blocks (never left & right).
          - Single blocks (aspect < 1.25 and area < 8500) are never split.
        """
        cname = detection.get("class_name")
        if cname not in ("red_block", "blue_block") or img is None:
            return [detection]

        # Suppress hand noise
        if not self.is_real_block(detection, img):
            return []

        bx, by, bw, bh = detection["bbox"]
        h_img, w_img = img.shape[:2]

        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        cw, ch = x2 - x1, y2 - y1
        if cw < 30 or ch < 30:
            return [detection]

        area = cw * ch
        aspect = max(cw, ch) / max(min(cw, ch), 1)

        # Geometrically, a single 2x2 Lego brick is roughly square (aspect < 1.30).
        # Two stacked or side-by-side blocks ALWAYS have aspect >= 1.35. NEVER split single blocks!
        if aspect < 1.32:
            return [detection]

        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

        # Horizontal seam in middle 30% to 70% of height
        r1, r2 = int(0.28 * ch), int(0.72 * ch)
        row_sums = np.sum(sobely, axis=1) if ch > 10 else np.array([0.0])
        best_r = r1 + int(np.argmax(row_sums[r1:r2])) if r2 > r1 else ch // 2
        r_peak = float(row_sums[best_r]) if r2 > r1 else 0.0
        r_local = float(np.mean(row_sums[max(0, best_r - 6):min(ch, best_r + 7)])) if ch > 0 else 1.0
        r_contrast = r_peak / max(r_local, 1e-3)
        r_global = float(np.mean(row_sums)) if len(row_sums) > 0 else 1.0
        r_ratio = r_peak / max(r_global, 1e-3)

        # Vertical seam in middle 30% to 70% of width
        c1, c2 = int(0.28 * cw), int(0.72 * cw)
        col_sums = np.sum(sobelx, axis=0) if cw > 10 else np.array([0.0])
        best_c = c1 + int(np.argmax(col_sums[c1:c2])) if c2 > c1 else cw // 2
        c_peak = float(col_sums[best_c]) if c2 > c1 else 0.0
        c_local = float(np.mean(col_sums[max(0, best_c - 6):min(cw, best_c + 7)])) if cw > 0 else 1.0
        c_contrast = c_peak / max(c_local, 1e-3)
        c_global = float(np.mean(col_sums)) if len(col_sums) > 0 else 1.0
        c_ratio = c_peak / max(c_global, 1e-3)

        split_axis = None
        split_pos = None

        if cw > ch:
            # Horizontal arrangement (side-by-side) cut vertically along X
            if (cw >= 1.35 * ch and (c_contrast >= 1.25 or c_ratio >= 1.20)) or cw >= 1.60 * ch:
                w1 = best_c
                w2 = cw - w1
                if min(w1, w2) / max(w1, w2) >= 0.30 and w1 >= 25 and w2 >= 25:
                    split_axis = "vertical"
                    split_pos = best_c
        elif ch > cw:
            # Vertical arrangement (stacked) cut horizontally along Y
            if (ch >= 1.35 * cw and (r_contrast >= 1.25 or r_ratio >= 1.20)) or ch >= 1.60 * cw:
                h1 = best_r
                h2 = ch - h1
                if min(h1, h2) / max(h1, h2) >= 0.30 and h1 >= 25 and h2 >= 25:
                    split_axis = "horizontal"
                    split_pos = best_r
        else:
            if c_peak >= r_peak and c_contrast >= 1.45 and aspect >= 1.35:
                split_axis = "vertical"
                split_pos = best_c
            elif r_peak > c_peak and r_contrast >= 1.45 and aspect >= 1.35:
                split_axis = "horizontal"
                split_pos = best_r

        conf = detection.get("confidence", 0.88)
        src = str(detection.get("source", "det")) + "_split"
        cls_id = detection["class_id"]

        # Execute vertical split (produces Left & Right blocks)
        if split_axis == "vertical" and split_pos is not None:
            x_cut = x1 + split_pos
            w1 = x_cut - bx
            w2 = (bx + bw) - x_cut
            if min(w1, w2) / max(w1, w2) < 0.40:
                w1 = cw // 2
                w2 = bw - w1
            if w1 >= 25 and w2 >= 25:
                return [
                    {
                        "class_name": cname,
                        "class_id": cls_id,
                        "bbox": (bx, by, w1, bh),
                        "center": (bx + w1 // 2, by + bh // 2),
                        "area": w1 * bh,
                        "confidence": float(conf),
                        "source": src,
                    },
                    {
                        "class_name": cname,
                        "class_id": cls_id,
                        "bbox": (bx + w1, by, w2, bh),
                        "center": (bx + w1 + w2 // 2, by + bh // 2),
                        "area": w2 * bh,
                        "confidence": float(conf),
                        "source": src,
                    },
                ]

        # Execute horizontal split (produces Top & Bottom blocks)
        elif split_axis == "horizontal" and split_pos is not None:
            y_cut = y1 + split_pos
            h1 = y_cut - by
            h2 = (by + bh) - y_cut
            if min(h1, h2) / max(h1, h2) < 0.40:
                h1 = ch // 2
                h2 = bh - h1
            if h1 >= 25 and h2 >= 25:
                return [
                    {
                        "class_name": cname,
                        "class_id": cls_id,
                        "bbox": (bx, by, bw, h1),
                        "center": (bx + bw // 2, by + h1 // 2),
                        "area": bw * h1,
                        "confidence": float(conf),
                        "source": src,
                    },
                    {
                        "class_name": cname,
                        "class_id": cls_id,
                        "bbox": (bx, by + h1, bw, h2),
                        "center": (bx + bw // 2, by + h1 + h2 // 2),
                        "area": bw * h2,
                        "confidence": float(conf),
                        "source": src,
                    },
                ]

        return [detection]

    def split_merged_red_block(self, detection, img):
        """Backward compatible helper forwarding to split_merged_block."""
        return self.split_merged_block(detection, img)

    def split_merged_blocks(self, detections, img):
        """
        Inspects all detections and splits any merged blocks (both red and blue blocks).
        """
        if not detections or img is None:
            return detections

        refined = []
        for d in detections:
            # Blue blocks covering both feet can be split here;
            # Red blocks are split with assembly context in ComponentDetector.corroborate_stacked_reds
            if d.get("class_name") == "blue_block":
                refined.extend(self.split_merged_block(d, img))
            else:
                refined.append(d)

        refined.sort(key=lambda d: d["area"], reverse=True)
        return refined

    def detect_yolo(self, img, imgsz=416):
        """Runs inference with the trained YOLO object detection model."""
        res = self.yolo_model.predict(img, imgsz=imgsz, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = res.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            cx, cy = bx + bw // 2, by + bh // 2
            area = bw * bh

            detections.append({
                "class_name": cls_name,
                "class_id": cls_id,
                "bbox": (bx, by, bw, bh),
                "center": (cx, cy),
                "area": area,
                "confidence": conf,
                "source": "yolo",
            })
        detections = [d for d in detections if self.is_real_block(d, img)]
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def detect(self, img):
        """
        Unified detection entry point.
        Uses YOLO detector if available, otherwise falls back to plastic-ratio color detector.
        Applies multi-block splitting to separate adjacent/stacked blocks of the same color.
        """
        if self.yolo_model is not None:
            raw_dets = self.detect_yolo(img)
        else:
            raw_dets = self.detect_color_plastic(img)

        return self.split_merged_blocks(raw_dets, img)

    def identify_incoming_object(self, detections):
        """
        Identifies an object presented by hand or held separately from the main cluster.
        Returns the detection dict of the candidate incoming block, or None.
        """
        if not detections or len(detections) <= 1:
            return None

        # Helper adjacency check
        def touches_other(box1, box2, gap=50):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
            dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
            return dx <= gap and dy <= gap

        # Find blocks that are NOT connected to any other block in the assembly
        # Only full physical blocks (not stud slivers) qualify as candidate incoming parts in hand
        isolated = []
        for i, d in enumerate(detections):
            bw, bh = d["bbox"][2], d["bbox"][3]
            area = d.get("area", bw * bh)
            if bw < 32 or bh < 32 or area < 1400:
                continue
            is_connected = any(touches_other(d["bbox"], other["bbox"]) for j, other in enumerate(detections) if j != i)
            if not is_connected:
                isolated.append(d)

        if not isolated:
            return None

        # From isolated blocks, pick the one furthest from the main assembly cluster
        def get_center(det):
            if "center" in det:
                return det["center"]
            bx, by, bw, bh = det["bbox"]
            return (bx + bw // 2, by + bh // 2)

        connected_boxes = [d for d in detections if d not in isolated]
        if connected_boxes:
            c_centers = np.array([get_center(d) for d in connected_boxes])
            cluster_center = np.median(c_centers, axis=0)
            dists = [np.hypot(get_center(d)[0] - cluster_center[0], get_center(d)[1] - cluster_center[1]) for d in isolated]
            max_idx = int(np.argmax(dists))
            if dists[max_idx] > 140:
                return isolated[max_idx]

        return None
