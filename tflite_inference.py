import os
import cv2
import numpy as np
import glob
import random
import argparse
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt

# Define function for inferencing with TFLite model and displaying results
def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='./results', txt_only=False):

    # Grab filenames of all images in test folder
    images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Randomly select test images
    images_to_test = random.sample(images, num_test_images)

    # Loop over every image and perform detection
    for image_path in images_to_test:

        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                detections.append([labels[int(classes[i])], scores[i], xmin, ymin, xmax, ymax])

        # Save detection results in .txt files
        if txt_only:
            image_fn = os.path.basename(image_path)
            base_fn, _ = os.path.splitext(image_fn)
            txt_result_fn = base_fn + '.txt'
            txt_savepath = os.path.join(savepath, txt_result_fn)

            with open(txt_savepath, 'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
    return


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFLite model inference on test images.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder containing images and labels.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .tflite model file.")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence threshold for detections.")
    parser.add_argument("--results_path", type=str, default='./results', help="Folder to save detection results.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to use for inference.")
    parser.add_argument("--txt_only", action='store_true', help="Only save results as text files without displaying images.")

    args = parser.parse_args()

    # Set up paths
    dataset_path = args.dataset_path
    model_path = args.model_path
    images_path = os.path.join(dataset_path, "valid")
    labels_path = os.path.join(dataset_path, "labelmap.txt")
    results_path = args.results_path

    # Check if results folder exists, if not, create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Get the list of images
    image_list = glob.glob(images_path + '/*.jpg') + glob.glob(images_path + '/*.JPG') + glob.glob(images_path + '/*.png') + glob.glob(images_path + '/*.bmp')
    num_test_images = min(len(image_list), args.num_images)

    # Run inferencing function
    print(f'Starting inference on {num_test_images} images...')
    tflite_detect_images(model_path, images_path, labels_path, args.min_conf, num_test_images, results_path, args.txt_only)
    print('Finished inferencing!')
