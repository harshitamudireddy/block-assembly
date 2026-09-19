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
    "state_0_unstarted",
    "state_1_blue_green",
    "state_2_red_attached",
    "state_3_yellow_attached",
    "state_4_blue2_attached",
    "state_5_mid_assembly",
    "state_6_red2_attached",
    "state_7_yellow2_attached",
    "state_8_complete",
]

# Human-readable step titles for HUD
STEP_TITLES = {
    "state_0_unstarted": "0. Unstarted / Parts Stage",
    "state_1_blue_green": "1. Blue + Green Base",
    "state_2_red_attached": "2. Red Side Block",
    "state_3_yellow_attached": "3. Yellow Front Block",
    "state_4_blue2_attached": "4. Second Blue Block",
    "state_5_mid_assembly": "5. Mid Joint Assembly",
    "state_6_red2_attached": "6. Second Red Block",
    "state_7_yellow2_attached": "7. Second Yellow Block",
    "state_8_complete": "8. Complete Assembly",
}

# Mapping of the 12 WhatsApp dataset videos to parts or states
VIDEO_MAPPING = {
    "WhatsApp Video 2026-09-19 at 19.08.02.mp4": {
        "type": "part",
        "target": "blue_block",
        "desc": "Single Blue Block in hand"
    },
    "WhatsApp Video 2026-09-19 at 19.08.03 (1).mp4": {
        "type": "part",
        "target": "red_block",
        "desc": "Single Red Block in hand"
    },
    "WhatsApp Video 2026-09-19 at 19.08.03 (2).mp4": {
        "type": "part",
        "target": "yellow_block",
        "desc": "Single Yellow Block in hand"
    },
    "WhatsApp Video 2026-09-19 at 19.08.03.mp4": {
        "type": "part",
        "target": "green_block",
        "desc": "Single Green Block in hand"
    },
    "WhatsApp Video 2026-09-19 at 19.08.37.mp4": {
        "type": "state",
        "target": "state_1_blue_green",
        "desc": "Step 1: Green block connected to Blue base"
    },
    "WhatsApp Video 2026-09-19 at 19.08.38 (1).mp4": {
        "type": "state",
        "target": "state_2_red_attached",
        "desc": "Step 2: Red block attached"
    },
    "WhatsApp Video 2026-09-19 at 19.08.38 (2).mp4": {
        "type": "state",
        "target": "state_3_yellow_attached",
        "desc": "Step 3: Yellow block attached to base"
    },
    "WhatsApp Video 2026-09-19 at 19.08.38 (3).mp4": {
        "type": "state",
        "target": "state_4_blue2_attached",
        "desc": "Step 4: Second blue block attached"
    },
    "WhatsApp Video 2026-09-19 at 19.08.38.mp4": {
        "type": "state",
        "target": "state_5_mid_assembly",
        "desc": "Step 5: Mid-assembly structure"
    },
    "WhatsApp Video 2026-09-19 at 19.08.39 (1).mp4": {
        "type": "state",
        "target": "state_6_red2_attached",
        "desc": "Step 6: Second red block attached"
    },
    "WhatsApp Video 2026-09-19 at 20.08.39 (2).mp4": {
        "type": "state",
        "target": "state_7_yellow2_attached",
        "desc": "Step 7: Second yellow block attached"
    },
    "WhatsApp Video 2026-09-19 at 19.08.39 (2).mp4": {
        "type": "state",
        "target": "state_7_yellow2_attached",
        "desc": "Step 7: Second yellow block attached"
    },
    "WhatsApp Video 2026-09-19 at 19.08.39.mp4": {
        "type": "state",
        "target": "state_8_complete",
        "desc": "Step 8: Fully completed 9-part block assembly"
    },
}

# Expected parts count per state
EXPECTED_PARTS_PER_STATE = {
    "state_0_unstarted": {"min_total": 0, "parts": {}},
    "state_1_blue_green": {"min_total": 2, "parts": {"blue_block": 1, "green_block": 1}},
    "state_2_red_attached": {"min_total": 3, "parts": {"blue_block": 1, "green_block": 1, "red_block": 1}},
    "state_3_yellow_attached": {"min_total": 4, "parts": {"blue_block": 1, "green_block": 1, "red_block": 1, "yellow_block": 1}},
    "state_4_blue2_attached": {"min_total": 5, "parts": {"blue_block": 2, "green_block": 1, "red_block": 1, "yellow_block": 1}},
    "state_5_mid_assembly": {"min_total": 6, "parts": {"blue_block": 2, "green_block": 1, "red_block": 1, "yellow_block": 1}},
    "state_6_red2_attached": {"min_total": 7, "parts": {"blue_block": 2, "green_block": 1, "red_block": 2, "yellow_block": 1}},
    "state_7_yellow2_attached": {"min_total": 8, "parts": {"blue_block": 2, "green_block": 1, "red_block": 2, "yellow_block": 2}},
    "state_8_complete": {"min_total": 8, "parts": {"blue_block": 2, "green_block": 1, "red_block": 2, "yellow_block": 2}},
}
