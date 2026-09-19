"""
Evaluation and benchmark script for Block Assembly Quality Inspection.
Evaluates:
  1. Detection accuracy and part counting on validation frames
  2. Assembly state sequence classification accuracy
  3. Spatial rule validation consistency

Usage:
    python evaluate.py
"""

import os
import cv2
from component_detector import ComponentDetector
from block_config import EXTRACTED_DIR, ASSEMBLY_STATES, STEP_TITLES


def evaluate_states(detector):
    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if not os.path.exists(states_dir):
        print(f"[ERROR] '{states_dir}' does not exist. Run extract_frames.py first.")
        return

    print("\n" + "=" * 70)
    print(" BENCHMARK: ASSEMBLY STATE CLASSIFICATION ON HELD-OUT VAL FRAMES")
    print("=" * 70)

    total_frames = 0
    correct_frames = 0
    state_results = {}

    for sname in sorted(os.listdir(states_dir)):
        s_path = os.path.join(states_dir, sname)
        if not os.path.isdir(s_path):
            continue

        val_files = [f for f in os.listdir(s_path) if f.startswith("vid_val_") and f.lower().endswith((".jpg", ".png"))]
        if not val_files:
            continue

        s_correct = 0
        s_total = len(val_files)

        for vf in val_files:
            fpath = os.path.join(s_path, vf)
            img = cv2.imread(fpath)
            if img is None:
                continue

            res = detector.analyze(img)
            pred = res["predicted_state"]

            if pred == sname:
                s_correct += 1

        acc = (s_correct / s_total) * 100.0 if s_total > 0 else 0.0
        state_results[sname] = {"correct": s_correct, "total": s_total, "acc": acc}

        total_frames += s_total
        correct_frames += s_correct

        print(f"  {sname:<28}: {s_correct:>3}/{s_total:<3} ({acc:>5.1f}%)")

    overall_acc = (correct_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
    print("-" * 70)
    print(f"  Overall Validation Accuracy: {correct_frames}/{total_frames} ({overall_acc:.1f}%)")
    print("=" * 70)


def evaluate_parts(detector):
    parts_dir = os.path.join(EXTRACTED_DIR, "parts")
    if not os.path.exists(parts_dir):
        return

    print("\n" + "=" * 70)
    print(" BENCHMARK: INDIVIDUAL OBJECT DETECTION ON ISOLATED PART VIDEOS")
    print("=" * 70)

    for pname in sorted(os.listdir(parts_dir)):
        p_path = os.path.join(parts_dir, pname)
        if not os.path.isdir(p_path):
            continue

        val_files = [f for f in os.listdir(p_path) if f.startswith("vid_val_") and f.lower().endswith((".jpg", ".png"))]
        if not val_files:
            continue

        det_correct = 0
        p_total = len(val_files)

        for vf in val_files:
            fpath = os.path.join(p_path, vf)
            img = cv2.imread(fpath)
            if img is None:
                continue

            res = detector.analyze(img)
            detections = res.get("detections", [])
            # Check if target part was detected
            if any(d["class_name"] == pname for d in detections):
                det_correct += 1

        acc = (det_correct / p_total) * 100.0 if p_total > 0 else 0.0
        print(f"  {pname:<28}: {det_correct:>3}/{p_total:<3} ({acc:>5.1f}%)")

    print("=" * 70)


def main():
    detector = ComponentDetector()
    evaluate_parts(detector)
    evaluate_states(detector)


if __name__ == "__main__":
    main()
