import numpy as np


class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lam = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n, d = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(d)

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                satisfies_constraint = y_[i] * (np.dot(self.w, x_i) + self.b) >= 1
                if satisfies_constraint:
                    self.w -= self.lr * (2 * self.lam * self.w)
                else:
                    self.w -= self.lr * (2 * self.lam * self.w - y_[i] * x_i)
                    self.b += self.lr * y_[i]

    def predict(self, X):
        y_hat = np.sign(np.dot(X, self.w) + self.b)
        return np.where(y_hat == 0, -1, 1)
