import sys
import os
import pickle
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import tree

from typing import Any

TRAIN_IMAGE_FOLDER  = "./train_images/"
TEXT_IMAGE_FOLDER   = "./text_images/"
SOURCE_IMAGES = "./source_images/"
MODELS_PATH = "./models/"

OPERATOR_HEIGHT = 13
OPERATOR_WIDTH  = 13

Array = Any
Model = tree.DecisionTreeClassifier

def slice_array_in_windows(a: Array, window_height: int, window_width: int) -> Array:
    b = sliding_window_view(a, (window_height, window_width))
    return b.reshape(b.shape[0] * b.shape[1], b.shape[2] * b.shape[3])

def trim_array_borders(a: Array, window_height: int, window_width: int) -> Array:
    h, w = a.shape
    horz_border = window_width//2
    vert_border = window_height//2
    b = a[vert_border:h-vert_border:, horz_border:w-horz_border:]
    return b

def input_image_from_path(f: str) -> Array:
   image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
   th, b_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   input_image = slice_array_in_windows(b_image, OPERATOR_HEIGHT, OPERATOR_WIDTH)

   return input_image

def output_image_from_path(f: str) -> Array:
   image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
   th, b_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   output_image = trim_array_borders(b_image, OPERATOR_HEIGHT, OPERATOR_WIDTH)

   return output_image.flatten()


def flatten_array(arr: Array, dimensions: list[int]) -> Array:
    shape = arr.shape
    dim = (
        [shape[i] for i in range(dimensions[0])]
        + [-1]
        + [shape[i] for i in range(dimensions[-1] + 1, len(shape)) ]
    )
    return arr.reshape(*dim)
    
def train_model(image_folder: str, text_image_folder: str) -> Model:
    image_files = [image_folder + f for f in os.listdir(image_folder)]
    text_image_files =  [text_image_folder + f for f in os.listdir(text_image_folder)]
    image_files.sort()
    text_image_files.sort()
    input_images = np.array([
        input_image_from_path(f)
        for f in image_files
    ])
    expected_output = np.array([
        output_image_from_path(f)
        for f in text_image_files
    ])

    input_images = flatten_array(input_images, [0, 1])
    expected_output = expected_output.flatten()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(input_images, expected_output)
    return clf

def reconstruct_image(arr: Array, height: int, width: int) -> Array:
    return arr.reshape(
        height - (OPERATOR_HEIGHT//2) * 2,
        width  - (OPERATOR_WIDTH//2) * 2,
    )

def predict_image(model: Model, image: Array) -> Array:
    h, w = image.shape
    X = slice_array_in_windows(image, OPERATOR_HEIGHT, OPERATOR_WIDTH)
    Y = model.predict(X)
    only_text_image = reconstruct_image(Y, h, w)
    only_text_image[only_text_image == 1] = 255
    return only_text_image

def train_new_model() -> None:
    model = train_model(TRAIN_IMAGE_FOLDER, TEXT_IMAGE_FOLDER)
    with open(MODELS_PATH + "11x11", "wb") as f:
        pickle.dump(model, f)

    print("SUCESS: model has been trained")
    

def predict(path: str) -> None:
    with open(MODELS_PATH + "11x11", "rb") as f:
        model = pickle.load(f)

    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    th, b_test_image = cv2.threshold(test_image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("Original", test_image)
    cv2.imshow("Only Text", predict_image(model, b_test_image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def usage() -> None:
    print("Usage:")
    print("    python main.py <command>")
    print("Commands:")
    print("    train")
    print("    extract <file-path>")
    print("Examples:")
    print("    python main.py train")
    print("    python main.py extract source_images/8.jpg")

def main() -> None:
    if len(sys.argv) < 2:
        usage()
        exit(1)

    command = sys.argv[1]
    if command == "train":
        train_new_model()
    elif command == "extract":
        if len(sys.argv) < 3:
            usage()
            exit(1)
        path = sys.argv[2]
        predict(path)

if __name__ == "__main__":
    main()
