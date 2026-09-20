"""
ComponentDetector: Unified inference coordinator for Block Assembly.
Combines:
  1. BlockDetector (YOLO / HSV Object Detection) -> localizes individual blocks & incoming parts
  2. AssemblyGraph (Spatial Rules & Graph Matching) -> verifies connections and sequence order
  3. Optional YOLO State Classifier -> corroborates global stage confidence
"""

import os
import cv2
from block_detector import BlockDetector, YOLO_DET_MODEL_PATH
from assembly_graph import AssemblyGraph
from block_config import ASSEMBLY_STATES, STEP_TITLES

YOLO_CLS_MODEL_PATH = "runs_classify/block_states/weights/best.pt"


class ComponentDetector:
    def __init__(
        self,
        det_model_path=YOLO_DET_MODEL_PATH,
        cls_model_path=YOLO_CLS_MODEL_PATH,
        conf_threshold=0.45,
    ):
        self.block_detector = BlockDetector(model_path=det_model_path, conf_threshold=conf_threshold)
        self.assembly_graph = AssemblyGraph()

        self.cls_model = None
        if os.path.exists(cls_model_path):
            try:
                from ultralytics import YOLO
                self.cls_model = YOLO(cls_model_path)
                print(f"[ComponentDetector] Loaded state classifier from '{cls_model_path}'")
            except Exception as e:
                print(f"[ComponentDetector] Could not load classifier ({e}).")

    def analyze(self, img_or_path, current_step_index=0):
        """
        Analyzes a single frame:
          - Detects all block components
          - Identifies incoming/held object
          - Verifies spatial assembly graph
          - Corroborates with classifier (if available)
        """
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        if img is None:
            return {
                "error": "Failed to read image",
                "predicted_state": "state0",
                "confidence": 0.0,
                "is_valid": False,
                "diagnostic": "Image load failure",
                "detections": [],
                "incoming_object": None,
            }

        # 1. Detect all blocks in frame
        detections = self.block_detector.detect(img)

        # 2. Identify incoming/held object
        incoming = self.block_detector.identify_incoming_object(detections)

        # 3. Evaluate spatial assembly graph
        target_state = ASSEMBLY_STATES[current_step_index] if current_step_index < len(ASSEMBLY_STATES) else None
        graph_eval = self.assembly_graph.evaluate(detections, current_target_step=target_state)

        inferred_state = graph_eval["inferred_state"]
        conf = graph_eval["confidence"]
        is_valid = graph_eval["is_valid"]
        diagnostic = graph_eval["diagnostic"]

        # 4. Corroborate with classification model if trained
        cls_pred = None
        cls_conf = 0.0
        if self.cls_model is not None:
            res_cls = self.cls_model.predict(img, verbose=False)[0]
            cls_pred = res_cls.names[res_cls.probs.top1]
            cls_conf = float(res_cls.probs.top1conf)

            # Blend confidences if classifier agrees
            if cls_pred == inferred_state:
                conf = (conf + cls_conf) / 2.0
            elif cls_conf > 0.85 and len(detections) <= 1:
                # Strong classifier override when objects are occluded
                inferred_state = cls_pred
                conf = cls_conf

        # 5. Check incoming object compatibility with next required step
        incoming_info = None
        if incoming is not None:
            incoming_cls = incoming["class_name"]
            next_idx = min(current_step_index + 1, len(ASSEMBLY_STATES) - 1)
            next_title = STEP_TITLES.get(ASSEMBLY_STATES[next_idx], "Next Step")

            # Determine expected block color for next step
            expected_block = None
            if next_idx == 1:
                expected_block = "green_block"
            elif next_idx == 2:
                expected_block = "blue_block"
            elif next_idx == 3:
                expected_block = "red_block"
            elif next_idx == 4:
                expected_block = "yellow_block"

            is_expected = (expected_block is None) or (incoming_cls == expected_block)
            incoming_info = {
                "class_name": incoming_cls,
                "confidence": incoming["confidence"],
                "bbox": incoming["bbox"],
                "is_expected": is_expected,
                "expected_block": expected_block,
                "message": f"Presenting: {incoming_cls} ({'CORRECT' if is_expected else 'WRONG PART for next step'})",
            }

        return {
            "predicted_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "detections": detections,
            "incoming_object": incoming_info,
            "part_counts": graph_eval["part_counts"],
            "spatial_checks": graph_eval["spatial_checks"],
            "signals": {
                "cls_pred": cls_pred,
                "cls_conf": cls_conf,
            },
        }
