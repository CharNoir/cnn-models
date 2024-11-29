import argparse
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

def bench_model(model_path, dataset_path, image_size, output_file, device):
    # Load the YOLO model
    model = YOLO(model_path, task='detect')

    # Run validation
    if device:
        benchmark(model=model_path, data=dataset_path, imgsz=image_size, device=device)
    else:
        benchmark(model=model_path, data=dataset_path, imgsz=image_size)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate YOLO model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--imgsz", type=int, required=True, help="Image size for validation")
    parser.add_argument("--output", type=str, required=False, default="validation_results.txt", help="Path to the output file")
    parser.add_argument("--device", type=str, help="Image size for validation")

    # Parse arguments
    args = parser.parse_args()

    # Call the validation function
    bench_model(args.model, args.dataset, args.imgsz, args.output, args.device)
