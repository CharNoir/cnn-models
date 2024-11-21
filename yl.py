from ultralytics import YOLO

# Load the exported TFLite Edge TPU model
edgetpu_model = YOLO("yolov5s-int8-224_edgetpu.tflite")

# Run inference
results = edgetpu_model("bus.jpg", imgsz=640)
print(results)

results = edgetpu_model("bus.jpg", imgsz=224)
print(results)