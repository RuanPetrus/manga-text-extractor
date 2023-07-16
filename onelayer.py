import sys
import os
import cv2
import numpy as np
from sklearn import tree
from numpy.lib.stride_tricks import sliding_window_view

from typing import Any

OPERATOR_HEIGHT = 13
OPERATOR_WIDTH  = 13

Model = tree.DecisionTreeClassifier
Array = Any

class OneLayer:
    def __init__(self, image_folder: str, text_image_folder: str):
        self.image_folder = image_folder
        self.text_image_folder = text_image_folder

    def train(self) -> None:
        image_files = [self.image_folder + f for f in os.listdir(self.image_folder)]
        text_image_files =  [self.text_image_folder + f for f in os.listdir(self.text_image_folder)]
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
        self.clf = clf

    def predict(self, image: Array) -> Array:
        h, w = image.shape
        X = slice_array_in_windows(image, OPERATOR_HEIGHT, OPERATOR_WIDTH)
        Y = self.clf.predict(X)
        only_text_image = reconstruct_image(Y, h, w)
        only_text_image[only_text_image == 1] = 255
        return only_text_image
        
def slice_array_in_windows(a: Array, window_height: int, window_width: int) -> Array:
    b = sliding_window_view(a, (window_height, window_width))
    return b.reshape(b.shape[0] * b.shape[1], b.shape[2] * b.shape[3])

def trim_array_borders(a: Array, window_height: int, window_width: int) -> Array:
    h, w = a.shape
    horz_border = window_width//2
    vert_border = window_height//2
    b = a[vert_border:h-vert_border:, horz_border:w-horz_border:]
    return b

def flatten_array(arr: Array, dimensions: list[int]) -> Array:
    shape = arr.shape
    dim = (
        [shape[i] for i in range(dimensions[0])]
        + [-1]
        + [shape[i] for i in range(dimensions[-1] + 1, len(shape)) ]
    )
    return arr.reshape(*dim)

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


def reconstruct_image(arr: Array, height: int, width: int) -> Array:
    return arr.reshape(
        height - (OPERATOR_HEIGHT//2) * 2,
        width  - (OPERATOR_WIDTH//2) * 2,
    )
