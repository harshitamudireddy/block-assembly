"""
Block Detector module: Localizes and identifies individual blocks.
Supports both:
  1. Color-space HSV segmentation (works out of the box, zero training required)
  2. Deep learning YOLOv8 object detection (when weights are available in runs_detect/)
"""

import os
import cv2
import numpy as np
from block_config import BLOCK_CLASSES, BLOCK_COLORS_BGR

YOLO_DET_MODEL_PATH = "runs_detect/block_detector/weights/best.pt"

# HSV ranges calibrated for the toy blocks
HSV_RANGES = {
    "blue_block": [
        (np.array([90, 60, 40]), np.array([135, 255, 255])),
    ],
    "green_block": [
        (np.array([35, 60, 40]), np.array([85, 255, 255])),
    ],
    "red_block": [
        (np.array([0, 70, 50]), np.array([10, 255, 255])),
        (np.array([165, 70, 50]), np.array([180, 255, 255])),
    ],
    "yellow_block": [
        (np.array([18, 80, 80]), np.array([34, 255, 255])),
    ],
}


class BlockDetector:
    def __init__(self, model_path=YOLO_DET_MODEL_PATH, conf_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.yolo_model = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                print(f"[BlockDetector] Loaded trained YOLO model from '{model_path}'")
            except Exception as e:
                print(f"[BlockDetector] Could not load YOLO model ({e}). Using HSV detector fallback.")
        else:
            print("[BlockDetector] No trained YOLO weights found. Using HSV color-segmentation detector.")

    def detect_hsv(self, img, min_area=700):
        """
        Detects blocks via HSV color thresholding + contour filtering.
        Returns a list of dicts: {class_name, class_id, bbox: (x, y, w, h), center: (cx, cy), area, confidence}
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections = []

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        for cls_name, ranges in HSV_RANGES.items():
            cls_id = BLOCK_CLASSES.index(cls_name)
            mask = np.zeros((h, w), dtype=np.uint8)

            for lower, upper in ranges:
                sub_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, sub_mask)

            # Morphological cleaning
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= min_area:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cx, cy = bx + bw // 2, by + bh // 2
                    
                    # Aspect ratio check to eliminate long skin/table artifacts
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect > 4.5:
                        continue

                    # Confidence estimation based on color purity and contour solidity
                    hull = cv2.convexHull(c)
                    solidity = float(area) / max(cv2.contourArea(hull), 1)
                    conf = min(0.98, max(0.50, 0.40 + 0.55 * solidity))

                    detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "bbox": (bx, by, bw, bh),
                        "center": (cx, cy),
                        "area": area,
                        "confidence": float(conf),
                        "source": "hsv",
                    })

        # Sort detections by area (largest first)
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def detect_yolo(self, img):
        """Runs inference with the trained YOLO object detection model."""
        res = self.yolo_model.predict(img, conf=self.conf_threshold, verbose=False)[0]
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
        return detections

    def detect(self, img):
        """
        Unified detection entry point.
        Uses YOLO detector if available, otherwise falls back to HSV color segmentation.
        """
        if self.yolo_model is not None:
            return self.detect_yolo(img)
        return self.detect_hsv(img)

    def identify_incoming_object(self, detections, previous_detections=None):
        """
        Identifies an object recently brought into the frame or held separately from
        the main cluster.
        Returns the detection dict of the candidate incoming block, or None.
        """
        if not detections:
            return None

        # If only 1 object in view, that single object is the focus
        if len(detections) == 1:
            return detections[0]

        # If multiple objects, identify the isolated/outer block (farthest from the assembly center of mass)
        centers = np.array([d["center"] for d in detections])
        median_center = np.median(centers, axis=0)

        # Calculate distances to center of mass
        dists = [np.hypot(c[0] - median_center[0], c[1] - median_center[1]) for c in centers]
        max_dist_idx = int(np.argmax(dists))

        # If the farthest block is significantly separated (> 120 px), it is an incoming part
        if dists[max_dist_idx] > 110:
            return detections[max_dist_idx]

        return None
