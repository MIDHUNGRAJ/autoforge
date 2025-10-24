import numpy as np
from autoforge.base import BaseEstimator, require_fit


class ElasticNet(BaseEstimator):
    def __init__(
        self,
        fit_intercept=True,
        max_iter=1000,
        learning_rate=0.1,
        l1_ratio=0.1,
        alpha=0.2,
        tol=1e-4,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.tol = tol
        self.fit_intercept = fit_intercept

    def soft_threshold(self, z, alpha):
        if z > alpha:
            return z - alpha
        elif z < -alpha:
            return z + alpha
        else:
            return 0

    def fit(self, X, y):
        m, n = X.shape
        self.coef_ = np.zeros(n)
        self.intercept_ = 0
        self.coef_old = self.coef_.copy()

        self.intercept_ = np.mean(y - X @ self.coef_)
        for _ in range(self.max_iter):
            for j in range(n):
                residual = y - (X @ self.coef_) + X[:, j] * self.coef_[j]
                rho = X[:, j] @ residual / m
                self.coef_[j] = (
                    self.soft_threshold(rho, self.alpha * self.l1_ratio)
                    / (1 + self.alpha)
                    * (1 - self.l1_ratio)
                )

            if np.max(np.abs(self.coef_ - self.coef_old)) < self.tol:
                break

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return self.coef_ @ X
