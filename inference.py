import argparse
import os
import time
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image
from sklearn.metrics import precision_score, recall_score, average_precision_score

def load_annotations(annotation_path, image_shape):
    """Load YOLO annotations for ground truth boxes."""
    ground_truths = []
    with open(annotation_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            box = list(map(float, parts[1:]))
            x_center, y_center, width, height = box
            x_min = (x_center - width / 2) * image_shape[1]
            y_min = (y_center - height / 2) * image_shape[0]
            x_max = (x_center + width / 2) * image_shape[1]
            y_max = (y_center + height / 2) * image_shape[0]
            ground_truths.append((class_id, x_min, y_min, x_max, y_max))
    return ground_truths

def run_inference(interpreter, image, threshold=0.5):
    """Perform inference and return detected boxes and scores."""
    common.set_input(interpreter, image)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    detections = detect.get_objects(interpreter, threshold)
    results = []
    for det in detections:
        results.append({
            "id": det.id,
            "bbox": (det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax),
            "score": det.score,
        })
    return results, inference_time

def calculate_metrics(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Calculate precision, recall, and average precision."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0  # Perfect match
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0  # No match

    # Convert to NumPy arrays for IoU calculation
    gt_array = np.array([box[1:] for box in gt_boxes])
    pred_array = np.array([box["bbox"] for box in pred_boxes])
    pred_scores = np.array(pred_scores)

    # Calculate IoU
    ious = compute_iou_matrix(gt_array, pred_array)
    matches = ious > iou_threshold

    # Assign matches
    tp = np.sum(matches, axis=1) > 0
    fp = ~tp
    precision = precision_score(tp, fp, zero_division=1)
    recall = recall_score(tp, fp, zero_division=1)
    ap = average_precision_score(tp, pred_scores)

    return precision, recall, ap

def compute_iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix for ground truth and predicted boxes."""
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt, pred)
    return iou_matrix

def compute_iou(box1, box2):
    """Compute IoU for two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def process_dataset(dataset_path, model_path, labels_path, iou_threshold=0.5):
    """Run inference on a dataset and calculate metrics."""
    # Load the model
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    label_map = dataset.read_label_file(labels_path)

    all_precisions = []
    all_recalls = []
    all_aps = []
    total_time = 0.0

    # Process each image
    for image_file in os.listdir(os.path.join(dataset_path, "images")):
        image_path = os.path.join(dataset_path, "images", image_file)
        annotation_path = os.path.join(dataset_path, "labels", image_file.replace(".jpg", ".txt"))

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        size = common.input_size(interpreter)
        image_resized = image.resize(size, Image.ANTIALIAS)

        # Load ground truth annotations
        gt_boxes = load_annotations(annotation_path, image.size)

        # Run inference
        pred_boxes, pred_scores, inference_time = run_inference(interpreter, image_resized)
        total_time += inference_time

        # Calculate metrics
        precision, recall, ap = calculate_metrics(gt_boxes, pred_boxes, pred_scores, iou_threshold)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_aps.append(ap)

    # Calculate mean metrics
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    map50_95 = np.mean(all_aps)

    print(f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, mAP@[50-95]: {map50_95:.4f}")
    print(f"Average Inference Time: {total_time / len(all_precisions):.4f} seconds per image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference and evaluate metrics on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the TFLite model file")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels txt file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder in YOLO format")
    args = parser.parse_args()

    process_dataset(args.dataset, args.model, args.labels)
