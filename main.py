import sys
import os
import pickle
import cv2
from onelayer import OneLayer
from multilayer import MultiLayer

from typing import Any

TRAIN_IMAGE_FOLDER  = "./train_images/"
TEXT_IMAGE_FOLDER   = "./text_images/"
SOURCE_IMAGES = "./source_images/"
MODELS_PATH = "./models/"


def train_new_model() -> None:
   if not os.path.exists(MODELS_PATH):
      os.mkdir(MODELS_PATH)
   model = MultiLayer(TRAIN_IMAGE_FOLDER, TEXT_IMAGE_FOLDER)
   model.train()
   with open(MODELS_PATH + "multilayer_13x13", "wb") as f:
      pickle.dump(model, f)

   print("SUCESS: model has been trained")


def predict(path: str) -> None:
    with open(MODELS_PATH + "multilayer_13x13", "rb") as f:
        model = pickle.load(f)

    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    th, b_test_image = cv2.threshold(test_image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    only_text_image = model.predict(b_test_image)
    cv2.imwrite("only_text.png", only_text_image)

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

    else:
       usage()
       exit(1)
       

if __name__ == "__main__":
    main()
