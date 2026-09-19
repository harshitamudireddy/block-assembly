# Block Assembly Quality Inspection System

An automated Computer Vision & Deep Learning quality inspection system for multi-part block assembly. It combines real-time **Object Detection** (localizing individual blocks and identifying incoming parts in hand) with an **Assembly Graph Rule Engine** (verifying physical connections, alignments, and sequential order) and **Temporal Consensus Smoothing**.

---

## Architecture: Hybrid Perception + Spatial Graph

```
Raw Camera / Video Frame
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│              BlockDetector (YOLOv8-detect / HSV)          │
│  - Detects all blocks: Blue, Green, Red, Yellow          │
│  - Bounding boxes [x, y, w, h], center coordinates       │
│  - Incoming part tracking (identifies object in hand)    │
└──────────────────────────────────────────────────────────┘
          │                                  │
          │ Bounding Boxes & Parts           │ Incoming Part Info
          ▼                                  ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│   Assembly Graph Rule Engine     │   │      Incoming Object Checker     │
│   (assembly_graph.py)            │   │  - Verifies part color matches   │
│  - Checks physical adjacency     │   │    next required assembly step   │
│  - Validates block connections   │   │  - Alerts if wrong block is      │
│  - Pinpoints exact joint defects │   │    brought into frame            │
└──────────────────────────────────┘   └──────────────────────────────────┘
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
          ┌──────────────────────────────────┐
          │    AssemblyStateMachine          │
          │  - Rolling-window consensus      │
          │  - Strict sequential advance     │
          │  - Skipped step error latching   │
          └──────────────────────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │     Industrial Inspection HUD    │
          │  - Bounding boxes & part labels  │
          │  - PASS / HOLD / REJECT banner   │
          │  - 8-Step sequential checklist   │
          └──────────────────────────────────┘
```

---

## Assembly Stages

| Stage | Name | Description | Required Parts |
| :---: | :--- | :--- | :--- |
| **0** | `state_0_unstarted` | Workspace empty / presenting parts | None |
| **1** | `state_1_blue_green` | Base joint: Green attached to Blue base | 1 Blue, 1 Green |
| **2** | `state_2_red_attached` | Red block attached to Green/Blue | 1 Blue, 1 Green, 1 Red |
| **3** | `state_3_yellow_attached` | Yellow block connected to base | 1 Blue, 1 Green, 1 Red, 1 Yellow |
| **4** | `state_4_blue2_attached` | Second Blue block attached to joint | 2 Blue, 1 Green, 1 Red, 1 Yellow |
| **5** | `state_5_mid_assembly` | Mid-assembly structure | 2 Blue, 1 Green, 1 Red, 1 Yellow |
| **6** | `state_6_red2_attached` | Second Red block attached | 2 Blue, 1 Green, 2 Red, 1 Yellow |
| **7** | `state_7_yellow2_attached`| Second Yellow block attached | 2 Blue, 1 Green, 2 Red, 2 Yellow |
| **8** | `state_8_complete` | Complete 9-part block figure | 2 Blue, 1 Green, 2 Red, 2 Yellow |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Video Frames
Extracts frames from the 12 videos in `DATASET/` with blur filtering, block visibility check, and an 80/20 temporal train/val split:
```bash
python extract_frames.py
```

### 3. Prepare Datasets & Auto-Annotate
Generates:
- `yolo_dataset_det/`: Object detection dataset with auto-generated YOLO bounding box labels
- `yolo_dataset_cls/`: State classification dataset split into stage folders
```bash
python prepare_dataset.py
```

### 4. Train Models
Train the YOLO Object Detector:
```bash
python train_detector.py --epochs 60 --model yolov8s.pt
```

*(Optional)* Train the Whole-Scene State Classifier:
```bash
python train_classifier.py --epochs 60
```

### 5. Run Live Inspection HUD
Launch the real-time HUD with webcam:
```bash
python live_demo.py
```
Or test on one of your recorded dataset videos:
```bash
python live_demo.py --video "DATASET/WhatsApp Video 2026-09-19 at 19.08.38 (1).mp4"
```
Or inspect a single image:
```bash
python live_demo.py --image "extracted_frames/states/state_2_red_attached/vid_val_000100.jpg"
```

### Controls in Live Mode:
- **`r`**: Reset sequence tracker back to Step 0.
- **`q`**: Quit the live inspection window.

---

## Unified Interactive Menu
You can also launch everything via the interactive CLI:
```bash
python main.py
```
