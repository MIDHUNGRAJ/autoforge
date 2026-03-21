import numpy as np
from collections import Counter
from autoforge.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    def __init__(self, max_depth=10, num_trees=10, min_samples_split=2):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )

            X_sample, y_sample = self.bootstrap_samples(x, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self, x, y):
        num_samples = x.shape[0]
        idx = np.random.choice(num_samples, num_samples, replace=True)
        return x[idx], y[idx]

    def most_common_label(self, y):
        common = Counter(y).most_common(1)[0][0]
        return common

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_preds])
        return predictions


class RandomForestRegressor:
    def __init__(self, max_depth=10, num_trees=10, min_samples_split=2):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )

            X_sample, y_sample = self.bootstrap_samples(x, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self, x, y):
        num_samples = x.shape[0]
        idx = np.random.choice(num_samples, num_samples, replace=True)
        return x[idx], y[idx]

    def most_common_label(self, y):
        common = Counter(y).most_common(1)[0][0]
        return common

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([np.mean(pred) for pred in tree_preds])
        return predictions
