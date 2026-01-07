import numpy as np
from autoforge.utils import sigmoid
from autoforge.base import BaseEstimator, require_fit


class SGDClassifier(BaseEstimator):
    """ """

    def __init__(self, learning_rate=0.01, max_iter=100, batch_size=32):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.max_iter):
            idx = np.random.permutation(n_samples)
            X_batch = X[idx]
            y_batch = y[idx]

            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                Xi = X_batch[i:end_idx]
                yi = y_batch[i:end_idx]

                y_pred = Xi @ self.weight + self.bias

                y_pred = sigmoid(y_pred)

                grad_w = (2 / len(Xi)) * Xi.T @ (y_pred - yi)
                grad_b = (2 / len(Xi)) * np.sum(y_pred - yi)

                self.weight -= self.lr * grad_w
                self.bias -= self.lr * grad_b

        self._mark_fitted()
        return self

    @require_fit
    def predict_prob(self, X):
        return sigmoid(X @ self.weight + self.bias)

    @require_fit
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
