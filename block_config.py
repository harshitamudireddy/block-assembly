"""
Central configuration for Block Assembly Inspection System.
Defines video mapping, block classes, assembly states, and spatial graph constraints.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "DATASET")
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
DET_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_det")
CLS_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_cls")

# Block Classes for Object Detection
BLOCK_CLASSES = [
    "blue_block",    # 0
    "green_block",   # 1
    "red_block",     # 2
    "yellow_block",  # 3
]

BLOCK_COLORS_BGR = {
    "blue_block": (235, 130, 40),    # Blue in BGR
    "green_block": (40, 200, 40),    # Green in BGR
    "red_block": (40, 40, 235),      # Red in BGR
    "yellow_block": (40, 220, 240),  # Yellow in BGR
}

# Sequential Assembly States
ASSEMBLY_STATES = [
    "state0",
    "state1",
    "state2",
    "state3",
    "state4",
    "state5",
    "state6",
    "state7",
    "state8",
]

# Human-readable step titles for HUD
STEP_TITLES = {
    "state0": "0. Unstarted / Parts Stage",
    "state1": "1. Blue + Green Base",
    "state2": "2. Two Blue + Green",
    "state3": "3. Red Block Attached",
    "state4": "4. Yellow Block Attached",
    "state5": "5. Stage 5 Assembly",
    "state6": "6. Stage 6 Assembly",
    "state7": "7. Stage 7 Assembly",
    "state8": "8. Complete Assembly",
}

# Mapping of dataset videos to parts or states according to their video names
VIDEO_MAPPING = {
    "blue_block.mp4": {
        "type": "part",
        "target": "blue_block",
        "desc": "Single Blue Block in hand",
    },
    "green_block.mp4": {
        "type": "part",
        "target": "green_block",
        "desc": "Single Green Block in hand",
    },
    "red_block.mp4": {
        "type": "part",
        "target": "red_block",
        "desc": "Single Red Block in hand",
    },
    "yellow_block.mp4": {
        "type": "part",
        "target": "yellow_block",
        "desc": "Single Yellow Block in hand",
    },
    "state1.mp4": {
        "type": "state",
        "target": "state1",
        "desc": "Step 1: Green block connected to Blue base",
    },
    "state2.mp4": {
        "type": "state",
        "target": "state2",
        "desc": "Step 2: Second Blue block attached to Green block",
    },
    "state3.mp4": {
        "type": "state",
        "target": "state3",
        "desc": "Step 3: Red block attached to Green block",
    },
    "state4.mp4": {
        "type": "state",
        "target": "state4",
        "desc": "Step 4: Yellow block attached to assembly",
    },
    "state5.mp4": {
        "type": "state",
        "target": "state5",
        "desc": "Step 5: Intermediate assembly stage",
    },
    "state6.mp4": {
        "type": "state",
        "target": "state6",
        "desc": "Step 6: Secondary blocks attached",
    },
    "state7.mp4": {
        "type": "state",
        "target": "state7",
        "desc": "Step 7: Sub-assembly near completion",
    },
    "state8.mp4": {
        "type": "state",
        "target": "state8",
        "desc": "Step 8: Fully completed assembly",
    },
}

# Expected parts count per state
EXPECTED_PARTS_PER_STATE = {
    "state0": {"min_total": 0, "parts": {}},
    "state1": {"min_total": 2, "parts": {"blue_block": 1, "green_block": 1}},
    "state2": {"min_total": 3, "parts": {"blue_block": 2, "green_block": 1}},
    "state3": {"min_total": 4, "parts": {"blue_block": 2, "green_block": 1, "red_block": 1}},
    "state4": {"min_total": 5, "parts": {"blue_block": 2, "green_block": 1, "red_block": 1, "yellow_block": 1}},
    "state5": {"min_total": 5, "parts": {"blue_block": 2, "green_block": 1}},
    "state6": {"min_total": 6, "parts": {"blue_block": 2, "green_block": 1}},
    "state7": {"min_total": 7, "parts": {"blue_block": 2, "green_block": 1}},
    "state8": {"min_total": 8, "parts": {"blue_block": 2, "green_block": 1}},
}
