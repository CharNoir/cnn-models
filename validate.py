import argparse
from ultralytics import YOLO

def validate_model(model_path, dataset_path, image_size, output_file):
    try:
        # Load the YOLO model
        model = YOLO(model_path, task='detect')

        # Run validation
        metrics = model.val(data=dataset_path, imgsz=image_size)

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

        # Append results to the output file
        with open(output_file, "a") as f:
            f.write(results)
    except Exception as e:
        error_message = f"Error during validation for model {model_path}: {e}\n"
        print(error_message)
        with open(output_file, "a") as f:
            f.write(error_message)
        exit(1)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate YOLO model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--imgsz", type=int, required=True, help="Image size for validation")
    parser.add_argument("--output", type=str, required=False, default="validation_results.txt", help="Path to the output file")

    # Parse arguments
    args = parser.parse_args()

    # Call the validation function
    validate_model(args.model, args.dataset, args.imgsz, args.output)
