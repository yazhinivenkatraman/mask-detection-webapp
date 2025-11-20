import os

image_path = "C:/Users/nsach/OneDrive/Desktop/meng_project/source"

# === Class definitions ===
CLASSES = ["incorrect_mask", "with_mask", "with_n95", "without_mask"]

# === Global constants ===
TARGET = 224

# === Visualization colors ===
COLORS = {
    "incorrect_mask": (0, 165, 255),  # orange
    "with_mask": (0, 255, 0),         # green
    "with_n95": (255, 255, 0),        # teal
    "without_mask": (0, 0, 255)       # red
}

# === Mappings ===
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# === Face detector paths ===
caffe_model = f"{image_path}/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
dep_prototxt = f"{image_path}/face_detection_model/deploy.prototxt"
