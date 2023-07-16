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

class MultiLayer:
    def __init__(self, image_folder: str, text_image_folder: str):
        self.image_folder = image_folder
        self.text_image_folder = text_image_folder
        
        self.first_layer_func = [
            slice_left_diagonal, slice_right_diagonal,
            slice_topleft, slice_topright,
            slice_bottonleft, slice_bottonright,
            slice_horz, slice_vert,
            slice_middle
        ]

    def train(self) -> None:
        image_files = [self.image_folder + f for f in os.listdir(self.image_folder)]
        text_image_files =  [self.text_image_folder + f for f in os.listdir(self.text_image_folder)]
        image_files.sort()
        text_image_files.sort()
        inputs = np.concatenate([
            input_image_from_path(f)
            for f in image_files
        ])

        expected_output = np.concatenate([
            output_image_from_path(f)
            for f in text_image_files
        ])

        expected_output = expected_output.flatten()
        first_layer_inputs = [
            f(inputs)
            for f in self.first_layer_func
        ]

        self.first_layer_ops = [
            tree.DecisionTreeClassifier().fit(op_input, expected_output)
            for op_input in first_layer_inputs
        ]

        second_layer_input = np.array([
            op.predict(op_input)
            for op, op_input in zip(self.first_layer_ops, first_layer_inputs)
        ]).T

        self.second_layer_op = tree.DecisionTreeClassifier().fit(second_layer_input, expected_output)

    def predict(self, image: Array) -> Array:
        h, w = image.shape
        X = slice_array_in_windows(image, OPERATOR_HEIGHT, OPERATOR_WIDTH)
        Y = self._predict(X)
        only_text_image = reconstruct_image(Y, h, w)
        only_text_image[only_text_image == 1] = 255
        return only_text_image

    def _predict(self, inputs: Array) -> Array:
        first_layer_inputs = [
            f(inputs)
            for f in self.first_layer_func
        ]
        
        second_layer_input = np.array([
            op.predict(op_input)
            for op, op_input in zip(self.first_layer_ops, first_layer_inputs)
        ]).T

        return self.second_layer_op.predict(second_layer_input)

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

def slice_array_in_windows(a: Array, window_height: int, window_width: int) -> Array:
    b = sliding_window_view(a, (window_height, window_width))
    return b.reshape(b.shape[0] * b.shape[1], b.shape[2], b.shape[3])

def slice_left_diagonal(arr: Array) -> Array:
    return np.concatenate([
        np.diagonal(arr, axis1=1, axis2=2, offset=off)
        for off in [-1, 0, 1]
    ], axis=1)

def slice_right_diagonal(arr: Array) -> Array:
    return slice_left_diagonal(arr[:, :, ::-1])

def ceil(a: int, b: int) -> int:
    return (a+b-1)//b

def slice_topleft(arr: Array) -> Array:
    _, h, w = arr.shape
    wh = ceil(h,2)
    ww = ceil(w,2)
    return flatten_array(arr[:, 0:wh+1, 1:ww+1], [1, 2])

def slice_topright(arr: Array) -> Array:
    _, h, w = arr.shape
    wh = ceil(h,2)
    ww = ceil(w,2)
    return flatten_array(arr[:, 0:wh+1, w-(ww+1):-1], [1, 2])
    
def slice_bottonleft(arr: Array) -> Array:
    _, h, w = arr.shape
    wh = ceil(h,2)
    ww = ceil(w,2)
    return flatten_array(arr[:, h-(wh+1):, 1:ww+1], [1, 2])

def slice_bottonright(arr: Array) -> Array:
    _, h, w = arr.shape
    wh = ceil(h,2)
    ww = ceil(w,2)
    return flatten_array(arr[:, h-(wh+1):, w-(ww+1):-1], [1, 2])

def slice_horz(arr: Array) -> Array:
    _, h, _ = arr.shape
    s = h//3
    start = (h-s)//2
    return flatten_array(arr[:, start:start+s, :], [1, 2])

def slice_vert(arr: Array) -> Array:
    _, _, w = arr.shape
    s = w//3
    start = (w-s)//2
    return flatten_array(arr[:, :, start:start+s], [1, 2])

def slice_middle(arr: Array) -> Array:
    _, h, w = arr.shape
    fh = ceil(h, 10)
    fw = ceil(w, 10)
    return flatten_array(arr[:, fh:-fh, fw:-fw], [1, 2])

if __name__ == "__main__":
   arr = np.array([
       [
        [ 1, 2, 3, 4, 5, 6],
        [ 1, 2, 3, 4, 5, 9],
        [ 1, 2, 3, 4, 5, 6],
        [ 1, 2, 3, 5, 7, 6],
       ],
       [
        [ 1, 2, 3, 4, 5, 6],
        [ 1, 2, 3, 4, 5, 9],
        [ 1, 2, 3, 4, 5, 6],
        [ 1, 2, 3, 5, 7, 6],
       ],
   ]) 

   inputs = slice_array_in_windows(arr, 3, 3)



   
