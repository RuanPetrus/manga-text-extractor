# Manga text extractor
This extract text from manga pages based on the [paper](https://dl.acm.org/doi/10.1145/3011549.3011560).

# Requirements
This project was created using python version 3.11.
And the following dependencies

``` sh
pip install numpy
pip install opencv-python
pip install matplotlib
pip install -U scikit-learn
```

# Usage
The default model is the multilayer 13x13 berserk. So it only extract Berserk images.

In order to extract the text from a image you can run
``` sh
python main.py extract <image-path>
python main.py extract ./source_images/berserk/6.jpg
```

# Changing the model
The models are located in the _models_ folder.  
If you want to change the model you can change the following line in _predict_ in the main file.
``` python
def predict(path: str) -> None:
    with open(MODELS_PATH + "<your-model>", "rb") as f:
        model = pickle.load(f)
    ...
```
Example:
``` python
def predict(path: str) -> None:
    with open(MODELS_PATH + "multilayer_13x13_onepiece", "rb") as f:
        model = pickle.load(f)
    ...
```

This will only works for models with 13x13 operators, if you want to change the size of operators you will have to change the following lines in the _multilayer_ file:

``` python
...
OPERATOR_HEIGHT = 13
OPERATOR_WIDTH  = 13
...
```

# TODO
- Neural nets?
- Support vector machines ?
- Better way to abstract and store models
- Refactor array code
