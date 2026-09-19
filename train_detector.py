"""
Trains a YOLOv8 Object Detection model on the annotated block dataset.
Detects all 4 block types (blue_block, green_block, red_block, yellow_block) simultaneously.

Usage:
    python train_detector.py
    python train_detector.py --epochs 60 --model yolov8s.pt
"""

import os
import argparse
import torch
from ultralytics import YOLO

from block_config import DET_DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Object Detector for Blocks")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO model (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    yaml_path = os.path.join(DET_DATASET_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[ERROR] '{yaml_path}' not found. Please run 'python prepare_dataset.py' first.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training YOLO Object Detector on device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if device == "cpu":
        print("Note: Running on CPU. For faster training, ensure CUDA-enabled PyTorch is installed.")

    model = YOLO(args.model)

    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        project=os.path.abspath("runs_detect"),
        name="block_detector",
        exist_ok=True,
    )

    print("\nTraining complete.")
    print("Best weights saved to: runs_detect/block_detector/weights/best.pt")
    print("Next step: Run 'python live_demo.py' to test detection and assembly checking.")


if __name__ == "__main__":
    main()
