import argparse
import os
import time
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image
from sklearn.metrics import average_precision_score


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
    """Perform inference and return detected boxes and inference time."""
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


def compute_iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix for ground truth and predicted boxes."""
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt, pred["bbox"])
    return iou_matrix


def calculate_tp_fp(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate True Positives (TP) and False Positives (FP) for a fixed IoU threshold."""
    tp = 0
    fp = 0
    assigned_gt = set()  # Track matched ground truths

    for pred in pred_boxes:
        matched = False
        for i, gt in enumerate(gt_boxes):
            if i in assigned_gt:
                continue  # Skip already matched ground truths
            iou = compute_iou(gt[1:], pred["bbox"])
            if iou >= iou_threshold:
                matched = True
                assigned_gt.add(i)
                break

        if matched:
            tp += 1
        else:
            fp += 1

    return tp, fp


def calculate_mAP(gt_boxes, pred_boxes, pred_scores, iou_thresholds):
    """Calculate mAP across multiple IoU thresholds."""
    aps = []

    for iou_threshold in iou_thresholds:
        # Match predictions to ground truths
        tp = []
        y_true = []
        y_score = []
        assigned_gt = set()

        for j, pred in enumerate(pred_boxes):
            matched = False
            for i, gt in enumerate(gt_boxes):
                if i in assigned_gt:
                    continue  # Skip already matched ground truths
                iou = compute_iou(gt[1:], pred["bbox"])
                if iou >= iou_threshold:
                    matched = True
                    assigned_gt.add(i)
                    break

            y_true.append(1 if matched else 0)
            y_score.append(pred_scores[j])
            tp.append(matched)

        # Add unmatched ground truths as false negatives
        y_true.extend([1] * (len(gt_boxes) - len(assigned_gt)))
        y_score.extend([0] * (len(gt_boxes) - len(assigned_gt)))

        # Calculate AP for the current IoU threshold
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

    # Calculate mAP
    mAP50 = aps[0]  # IoU = 0.5
    mAP50_95 = np.mean(aps)
    return mAP50, mAP50_95


def process_dataset(dataset_path, model_path, labels_path, iou_thresholds):
    """Run inference on a dataset and calculate TP, FP, and mAP."""
    # Load the model
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    label_map = dataset.read_label_file(labels_path)

    total_tp = 0
    total_fp = 0
    total_time = 0.0
    all_mAP50 = []
    all_mAP50_95 = []

    images = [f for f in os.listdir(os.path.join(dataset_path, "images")) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(images)

    # Process each image
    for idx, image_file in enumerate(images):
        image_path = os.path.join(dataset_path, "images", image_file)
        annotation_path = os.path.join(dataset_path, "labels", image_file.replace(".jpg", ".txt"))

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        size = common.input_size(interpreter)
        image_resized = image.resize(size, Image.LANCZOS)

        # Load ground truth annotations
        gt_boxes = load_annotations(annotation_path, image.size)

        # Run inference
        pred_boxes, inference_time = run_inference(interpreter, image_resized)
        total_time += inference_time

        pred_scores = [pred["score"] for pred in pred_boxes]

        # Calculate True Positives and False Positives
        tp, fp = calculate_tp_fp(gt_boxes, pred_boxes, iou_threshold=0.75)
        total_tp += tp
        total_fp += fp

        # Calculate mAP
        mAP50, mAP50_95 = calculate_mAP(gt_boxes, pred_boxes, pred_scores, iou_thresholds)
        all_mAP50.append(mAP50)
        all_mAP50_95.append(mAP50_95)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images [{100 * (idx + 1) / total_images:.1f}%]")

    # Calculate overall mAP
    mean_mAP50 = np.mean(all_mAP50)
    mean_mAP50_95 = np.mean(all_mAP50_95)

    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Precision: {total_tp/(total_tp+total_fp)}")
    print(f"Box(mAP@50): {mean_mAP50:.4f}, Box(mAP@[50-95]): {mean_mAP50_95:.4f}")
    print(f"Average Inference Time: {1000 * total_time / total_images:.1f} milliseconds per image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference and evaluate metrics on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the TFLite model file")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels txt file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder in YOLO format")
    args = parser.parse_args()

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # IoU thresholds from 0.50 to 0.95
    process_dataset(args.dataset, args.model, args.labels, iou_thresholds)
