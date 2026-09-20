"""
Assembly Graph & Spatial Rule Engine.
Evaluates detected block bounding boxes against the expected multi-part assembly graph,
verifying both component counts and relative spatial constraints (adjacency, alignment).
Provides diagnostic error messages pinpointing which joint/block is incorrect.
"""

import numpy as np
from block_config import EXPECTED_PARTS_PER_STATE, ASSEMBLY_STATES, STEP_TITLES


def compute_iou(boxA, boxB):
    """Computes Intersection over Union between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def are_adjacent(boxA, boxB, max_gap=80):
    """
    Checks if two bounding boxes touch or are closely adjacent (within max_gap pixels).
    """
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA

    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    # Horizontal distance between boxes
    dx = max(0, max(xA1 - xB2, xB1 - xA2))
    # Vertical distance between boxes
    dy = max(0, max(yA1 - yB2, yB1 - yA2))

    return dx <= max_gap and dy <= max_gap


class AssemblyGraph:
    def __init__(self):
        pass

    def evaluate(self, detections, current_target_step=None):
        """
        Evaluates a list of detections against the expected assembly sequence.
        Returns:
            {
                "inferred_state": state_name,
                "confidence": float,
                "is_valid": bool,
                "diagnostic": str,
                "part_counts": {class_name: count},
                "spatial_checks": list of dicts
            }
        """
        if not detections:
            return {
                "inferred_state": "state0",
                "confidence": 0.0,
                "is_valid": True,
                "diagnostic": "No blocks detected in workspace",
                "part_counts": {},
                "spatial_checks": [],
            }

        # 1. Count detected parts by class
        part_counts = {}
        parts_by_class = {}
        for d in detections:
            cname = d["class_name"]
            part_counts[cname] = part_counts.get(cname, 0) + 1
            if cname not in parts_by_class:
                parts_by_class[cname] = []
            parts_by_class[cname].append(d)

        total_blocks = len(detections)

        # 2. Check each assembly stage starting from most advanced down to base
        inferred_state = "state0"
        best_match_score = 0.0
        diagnostic = "Workspace active"
        is_valid = True
        spatial_checks = []

        # Evaluate against all states
        for state_name in reversed(ASSEMBLY_STATES):
            spec = EXPECTED_PARTS_PER_STATE.get(state_name, {})
            req_parts = spec.get("parts", {})
            min_tot = spec.get("min_total", 0)

            # Check if current detections satisfy the minimum required parts count
            has_required_counts = True
            missing_part = None
            for req_cls, req_cnt in req_parts.items():
                if part_counts.get(req_cls, 0) < req_cnt:
                    has_required_counts = False
                    missing_part = req_cls
                    break

            if has_required_counts and total_blocks >= min_tot:
                inferred_state = state_name
                break

        # 3. Spatial Relationship Checks for the Inferred State
        if inferred_state == "state1":
            blue_boxes = parts_by_class.get("blue_block", [])
            green_boxes = parts_by_class.get("green_block", [])
            if blue_boxes and green_boxes:
                adj = are_adjacent(blue_boxes[0]["bbox"], green_boxes[0]["bbox"])
                spatial_checks.append({"rule": "Blue-Green Adjacency", "passed": adj})
                if not adj:
                    is_valid = False
                    diagnostic = "ALIGNMENT: Green block not attached to Blue base block"
                else:
                    diagnostic = "PASS: Blue + Green base securely assembled"

        elif inferred_state == "state2":
            blue_boxes = parts_by_class.get("blue_block", [])
            green_boxes = parts_by_class.get("green_block", [])
            if len(blue_boxes) >= 2 and green_boxes:
                adj1 = are_adjacent(blue_boxes[0]["bbox"], green_boxes[0]["bbox"])
                adj2 = are_adjacent(blue_boxes[1]["bbox"], green_boxes[0]["bbox"])
                adj = adj1 and adj2
                spatial_checks.append({"rule": "Two Blue Blocks to Green Adjacency", "passed": adj})
                if not adj:
                    is_valid = False
                    diagnostic = "ALIGNMENT: Both Blue blocks must be attached to Green block"
                else:
                    diagnostic = "PASS: Two Blue blocks attached to Green block"

        elif inferred_state in ["state3", "state4", "state5"]:
            # Check cluster connectivity across all parts
            connected = True
            for i in range(len(detections) - 1):
                boxA = detections[i]["bbox"]
                # Must be adjacent to at least one other block
                has_adj = any(are_adjacent(boxA, detections[j]["bbox"]) for j in range(len(detections)) if j != i)
                if not has_adj:
                    connected = False
                    break
            spatial_checks.append({"rule": "Assembly Connectivity", "passed": connected})
            if not connected:
                is_valid = False
                diagnostic = "LOOSE PART: One or more blocks are detached from the assembly"
            else:
                diagnostic = f"PASS: {STEP_TITLES.get(inferred_state, inferred_state)} satisfied"

        elif inferred_state in ["state6", "state7", "state8"]:
            connected = True
            for i in range(len(detections)):
                boxA = detections[i]["bbox"]
                has_adj = any(are_adjacent(boxA, detections[j]["bbox"]) for j in range(len(detections)) if j != i)
                if not has_adj:
                    connected = False
                    break
            spatial_checks.append({"rule": "Complete Structure Integrity", "passed": connected})
            if not connected:
                is_valid = False
                diagnostic = "STRUCTURAL DEFECT: Blocks are detached or misaligned"
            else:
                diagnostic = "PASS: Complete block structure verified!"

        # Confidence: average detection confidence of participating parts
        conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

        return {
            "inferred_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "part_counts": part_counts,
            "spatial_checks": spatial_checks,
        }
