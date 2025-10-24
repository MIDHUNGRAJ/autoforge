import numpy as np
from autoforge.base import BaseEstimator, require_fit


class Lasso(BaseEstimator):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

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
        self.intercept_ = 0.0
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n):
                residual = y - (X @ self.coef_) + X[:, j] * self.coef_[j]
                rho = X[:, j] @ residual
                self.coef_[j] = self.soft_threshold(rho, self.alpha) / np.sum(
                    X[:, j] ** 2
                )

            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = y - (X @ self.coef_)

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_
