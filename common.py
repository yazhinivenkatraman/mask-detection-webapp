# Necessary imports and uploads

import cv2
import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
import time
import glob
import os
import pathlib
import re
import random
import imageio.v2 as iio
import shutil
import albumentations as A


from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNet, VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D
import tensorflow_datasets as tfds
from PIL import Image, UnidentifiedImageError


image_path = "C:/Users/nsach/OneDrive/Desktop/meng_project/source"
dataset_name = "dataset"
processed_folder = "processed"
dataset_root = os.path.join(image_path, dataset_name)
processed_root = os.path.join(image_path, processed_folder)

# === Class definitions ===
CLASSES = ["incorrect_mask", "with_mask", "with_n95", "without_mask"]

# === Global constants ===
# standard input size for MobileNetV2, VGG19, etc.
TARGET = 224

# === Optional visualization colors ===
COLORS = {
    "incorrect_mask": (0, 165, 255),  # orange
    "with_mask": (0, 255, 0),         # green
    "with_n95": (255, 255, 0),        # cyan-yellow
    "without_mask": (0, 0, 255)       # red
}

# === Mappings ===
# Helper: get class index or label quickly
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

caffe_model = f"{image_path}/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
dep_prototxt = f"{image_path}/face_detection_model/deploy.prototxt"