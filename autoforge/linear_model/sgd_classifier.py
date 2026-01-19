import numpy as np
from autoforge.utils import sigmoid
from autoforge.base import BaseEstimator, require_fit


class SGDClassifier(BaseEstimator):
    """ """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        batch_size=32,
        momentum=0.0,
        nesterov=True,
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.momentum = momentum  # setting momentum to zero, normal SGD
        self.nag = nesterov

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        self.v_w = np.zeros(n_features)
        self.v_b = 0

        for epoch in range(self.max_iter):
            idx = np.random.permutation(n_samples)
            X_batch = X[idx]
            y_batch = y[idx]

            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                Xi = X_batch[i:end_idx]
                yi = y_batch[i:end_idx]

                if self.nag:
                    weight_lookahead = self.weight - self.momentum * self.v_w
                    bias_lookahead = self.bias - self.momentum * self.v_b

                    y_pred = Xi @ weight_lookahead + bias_lookahead
                    y_pred = sigmoid(y_pred)

                else:
                    y_pred = Xi @ self.weight + self.bias
                    y_pred = sigmoid(y_pred)

                grad_w = (2 / len(Xi)) * Xi.T @ (y_pred - yi)
                grad_b = (2 / len(Xi)) * np.sum(y_pred - yi)

                self.v_w = self.momentum * self.v_w + self.lr * grad_w
                self.v_b = self.momentum * self.v_b + self.lr * grad_b

                self.weight -= self.v_w
                self.bias -= self.v_b

        self._mark_fitted()
        return self

    @require_fit
    def predict_prob(self, X):
        return sigmoid(X @ self.weight + self.bias)

    @require_fit
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
