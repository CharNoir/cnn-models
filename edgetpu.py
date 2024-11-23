# Since I cannot directly execute YOLO or Roboflow code in this environment, I will provide you with the code to run locally.

# Step 1: Ensure you have the necessary libraries installed
# pip install ultralytics roboflow numpy

# Step 2: Code to perform validation on the COCO dataset with YOLOv8n model for EdgeTPU

from ultralytics import YOLO
from roboflow import Roboflow
import os
import time

# Download the COCO dataset
#rf = Roboflow(api_key="29BoJlnW33DXMTyQnnLP")
#project = rf.workspace("microsoft").project("coco")
#version = project.version(34)
#dataset = version.download("yolov8")
#dataset_path = os.path.join(dataset.location, "data.yaml")

# Load the EdgeTPU optimized model
edgetpu_model = YOLO("yolov8n.pt")

# Validation function
def validate_model(model_path, dataset_path, image_size):
    model = YOLO(model_path)
    metrics = model.val(data=dataset_path, imgsz=image_size)
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Mean Precision: {metrics.box.mp}")
    print(f"Mean Recall: {metrics.box.mr}")
    print(f"Speed: {metrics.speed['inference']}")
    
models = [
    ["yolov8n_coco_320_edgetpu.tflite", "coco8.yaml", 320],
    ["yolov8n_coco_640_edgetpu.tflite", "coco8.yaml", 640],
    ["yolov8s_coco_320_edgetpu.tflite", "coco8.yaml", 320],
    ["yolov8s_coco_640_edgetpu.tflite", "coco8.yaml", 640],
    ["yolo11n_coco_320_edgetpu.tflite", "coco8.yaml", 320],  
    ["yolo11n_coco_640_edgetpu.tflite", "coco8.yaml", 640],
    ["yolo11s_coco_320_edgetpu.tflite", "coco8.yaml", 320],
    ["yolo11s_coco_640_edgetpu.tflite", "coco8.yaml", 640],
]
# Run validation
for model in models:
    print("...................................................")
    print(model[0])
    validate_model(model[0], model[1], model[2])
    print()