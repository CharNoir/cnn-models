from ultralytics import YOLO
import gc

# Validation function
def validate_model(model_path, dataset_path, image_size):
    model = YOLO(model_path)
    metrics = model.val(data=dataset_path, imgsz=image_size)
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Mean Precision: {metrics.box.mp}")
    print(f"Mean Recall: {metrics.box.mr}")
    print(f"Speed: {metrics.speed['inference']}")
    
    del model
    del metrics
    
models = [
    ["yolov8n_coco_320_edgetpu.tflite", "coco8.yaml", 320],
    ["yolov8n_coco_640_edgetpu.tflite", "coco8.yaml", 640],
    ["yolov8s_coco_320_edgetpu.tflite", "coco8.yaml", 320],
    ["yolov8s_coco_640_edgetpu.tflite", "coco8.yaml", 40],
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
    gc.collect()