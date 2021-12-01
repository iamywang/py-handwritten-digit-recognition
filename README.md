# Python-DBRHD-Digits-Recognition

Handwritten digits recognition using python.

## Dataset

- train dataset: `digits/trainingDigits`
- test dataset: `digits/testDigits`

## Python

- python: 3.9
- lib: numpy, sklearn, tensorflow, keras

## Algorithm

- KNN: using sklearn, `knn.py`
- MLP: using sklearn, `mlp.py`
- CNN: using tensorflow.keras, `cnn.py`

## KNN Example

output:

```shell
$ python knn.py
K=1, total：946, wrong：13, wrong rate：0.01374, correct rate：0.98626
K=3, total：946, wrong：12, wrong rate：0.01268, correct rate：0.98732
K=5, total：946, wrong：18, wrong rate：0.01903, correct rate：0.98097
K=7, total：946, wrong：22, wrong rate：0.02326, correct rate：0.97674
```
