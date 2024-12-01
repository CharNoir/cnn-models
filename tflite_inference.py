import os
import time
import argparse
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils import edgetpu, dataset
from sklearn.metrics import average_precision_score
from PIL import Image


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


def calculate_metrics(gt_boxes, pred_boxes, pred_scores, iou_thresholds):
    """Calculate precision, recall, and mAP across multiple IoU thresholds."""
    precisions = []
    recalls = []
    aps = []

    for iou_threshold in iou_thresholds:
        tp, fp = 0, 0
        assigned_gt = set()

        y_true = []
        y_score = []
        for pred in pred_boxes:
            matched = False
            for i, gt in enumerate(gt_boxes):
                if i in assigned_gt:
                    continue
                if pred[0] == gt[0]:  # Ensure class match
                    iou = compute_iou(gt[1:], pred[1])
                    if iou >= iou_threshold:
                        matched = True
                        assigned_gt.add(i)
                        break

            y_true.append(1 if matched else 0)
            y_score.append(pred[2])
            if matched:
                tp += 1
            else:
                fp += 1

        y_true.extend([1] * (len(gt_boxes) - len(assigned_gt)))
        y_score.extend([0] * (len(gt_boxes) - len(assigned_gt)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
        ap = average_precision_score(y_true, y_score) if y_true else 0

        precisions.append(precision)
        recalls.append(recall)
        aps.append(ap)

    mAP50 = aps[0]  # IoU = 0.5
    mAP50_95 = np.mean(aps)

    return max(precisions), max(recalls), mAP50, mAP50_95


def run_inference(dataset_path, model_path, labels_path, min_conf, iou_thresholds):
    """Run inference on images and calculate metrics."""
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_size = common.input_size(interpreter)
    labels = dataset.read_label_file(labels_path)

    image_dir = os.path.join(dataset_path, "images")
    annotation_dir = os.path.join(dataset_path, "annotations")

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    total_metrics = []
    total_inference_time = 0.0

    for idx, image_file in enumerate(images):
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, image_file.replace(".jpg", ".txt"))

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize(input_size, Image.LANCZOS)

        # Load ground truth annotations
        gt_boxes = []
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                box = list(map(float, parts[1:]))
                x_center, y_center, width, height = box
                x_min = (x_center - width / 2) * image.width
                y_min = (y_center - height / 2) * image.height
                x_max = (x_center + width / 2) * image.width
                y_max = (y_center + height / 2) * image.height
                gt_boxes.append([class_id, x_min, y_min, x_max, y_max])

        # Run inference
        common.set_input(interpreter, np.asarray(image_resized))
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        detections = detect.get_objects(interpreter, min_conf)
        pred_boxes = [
            [det.id, (det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax), det.score]
            for det in detections
        ]
        pred_scores = [x[2] for x in pred_boxes]

        # Calculate metrics
        precision, recall, mAP50, mAP50_95 = calculate_metrics(gt_boxes, pred_boxes, pred_scores, iou_thresholds)
        total_metrics.append((precision, recall, mAP50, mAP50_95))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(images):
            print(f"Processed {idx + 1}/{len(images)} images [{100 * (idx + 1) / len(images):.1f}%]")

    # Calculate overall metrics
    avg_precision = np.mean([x[0] for x in total_metrics])
    avg_recall = np.mean([x[1] for x in total_metrics])
    avg_mAP50 = np.mean([x[2] for x in total_metrics])
    avg_mAP50_95 = np.mean([x[3] for x in total_metrics])

    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'mAP@50: {avg_mAP50:.4f}')
    print(f'mAP@[50-95]: {avg_mAP50_95:.4f}')
    print(f'Average Inference Time: {total_inference_time / len(images):.4f} seconds per image')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyCoral inference and compute metrics.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder containing images and annotations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .tflite model file.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to label file.")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence threshold for detections.")
    args = parser.parse_args()

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # IoU thresholds from 0.50 to 0.95
    run_inference(args.dataset_path, args.model_path, args.labels_path, args.min_conf, iou_thresholds)
