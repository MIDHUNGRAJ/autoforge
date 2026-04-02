import numpy as np
from collections import Counter


# Classification
class KNeighborsClassifier:
    def __init__(self, metric="l2", k=3):
        self.metric = metric
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self.pred_(data) for data in X_test]
        return predictions

    def pred_(self, data):
        distance = (
            [self.l2_distance(data, x) for x in self.X_train]
            if self.metric == "l2"
            else [self.l1_distance(data, x) for x in self.X_train]
        )
        nearest_idx = np.argsort(distance)[: self.k]
        nearest_labels = [self.y_train[i] for i in nearest_idx]
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        return most_common_label

    def l1_distance(self, x1, x2):
        dist = np.sum(np.abs((x1 - x2)))
        return dist

    def l2_distance(self, x1, x2):
        dist = np.sqrt(np.sum((x1 - x2) ** 2))
        return dist


# Regression
class KNeighborsRegressor:
    def __init__(self, metric="l2", k=3):
        self.metric = metric
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self.pred_(data) for data in X_test]
        return predictions

    def pred_(self, data):
        distance = (
            [self.l2_distance(data, x) for x in self.X_train]
            if self.metric == "l2"
            else [self.l1_distance(data, x) for x in self.X_train]
        )
        nearest_idx = np.argsort(distance)[: self.k]
        nearest_labels = [self.y_train[i] for i in nearest_idx]
        most_common_label = np.mean(nearest_labels)
        return most_common_label

    def l1_distance(self, x1, x2):
        dist = np.sum(np.abs((x1 - x2)))
        return dist

    def l2_distance(self, x1, x2):
        dist = np.sqrt(np.sum((x1 - x2) ** 2))
        return dist
