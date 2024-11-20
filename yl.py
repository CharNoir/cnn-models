from ultralytics import YOLO

# Load the exported TFLite Edge TPU model
edgetpu_model = YOLO("yolov8n_full_integer_quant_edgetpu.tflite")

# Run inference
results = edgetpu_model(["bus.jpg", "bus.jpg"], imgsz=640)
print(results)

results = edgetpu_model(["bus.jpg", "bus.jpg"], imgsz=640)
print(results)