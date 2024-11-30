import argparse

import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

def inference(model_path, labels_path, image_path):
    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, model_path)
    label_file = os.path.join(script_dir, labels_path)
    image_file = os.path.join(script_dir, image_path)

    # Initialize the TF interpreter
    interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()

    # Resize the image
    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

    # Run an inference
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
  

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate YOLO model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the tflite model file")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels txt file")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the validation function
    inference(args.model, args.labels, args.image)