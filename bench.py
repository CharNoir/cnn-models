from ultralytics import YOLO

#benchmark(model="yolov8n.pt", data="coco8.yaml")

#benchmark(model="yolov8n.pt", data="coco.yaml")

def validate_model(model_path, dataset_path):
     # Load the YOLO model
    model = YOLO(model_path, task='detect')

        # Run validation
    metrics = model.val(data=dataset_path, imgsz=640)

        # Format results
    results = (
        f"Model: {model_path}\n"
        f"Dataset: {dataset_path}\n"
        f"mAP50-95: {metrics.box.map:.4f}\n"
        f"F1 score: {(2*metrics.box.mp*metrics.box.mr / (metrics.box.mp+metrics.box.mr)):.4f}\n"
        f"Mean Precision: {metrics.box.mp:.4f}\n"
        f"Mean Recall: {metrics.box.mr:.4f}\n"
        f"Speed (Inference): {metrics.speed['inference']:.2f} ms/image\n"
        "---------------------------------------------\n"
    )

        # Print results to console
    print(results)

        
validate_model("yolov8n.pt", "coco8.yaml")