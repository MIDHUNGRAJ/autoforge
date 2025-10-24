import numpy as np


def accuracy_score(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    return correct / len(y_true)
