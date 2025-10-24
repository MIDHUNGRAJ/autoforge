import numpy as np
from autoforge.base import BaseEstimator, require_fit


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression (L2 Regularization).

    This implementation solves the ridge regression problem:
        theta = (X^T X + alpha * I)^{-1} X^T y

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.

    fit_intercept : bool, default=True
        Whether to calculate and include an intercept term in the model.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term (intercept) in the linear model. Set to 0.0 if
        fit_intercept=False.
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        solver="solve",
        max_iter=1000,
        learning_rate=0.01,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.solver = solver
        self.beta = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def solve_(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        n_features = X.shape[1]
        I = np.eye(n_features)  
        I[0, 0] = 0

        x_t_x = X.T @ X
        XtX_lambd = x_t_x + self.alpha * I
        x_t_y = X.T @ y

        self.beta = np.linalg.solve(XtX_lambd, x_t_y)

        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

    def sgd_(self, X, y):
        n_samples, n_features = X.shape

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0 if self.fit_intercept else None

        for epoch in range(self.max_iter):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # Compute linear prediction
                y_pred = np.dot(xi, self.coef_)
                if self.fit_intercept:
                    y_pred += self.intercept_

                error = y_pred - yi

                # Update coefficients (with regularization)
                grad_w = error * xi + self.alpha * self.coef_
                self.coef_ -= self.learning_rate * grad_w

                # Update intercept separately (no regularization)
                if self.fit_intercept:
                    self.intercept_ -= self.learning_rate * error

    def fit(self, X, y):
        """
        Fit ridge regression model to training data using the closed-form
        solution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if self.solver == "solve":
            self.solve_(X, y)
        elif self.solver == "sgd":
            self.sgd_(X, y)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        if self.solver == "solve":
            if self.fit_intercept:
                X = np.c_[np.ones(X.shape[0]), X]

            return X @ self.beta
        else:
            y_pred = np.dot(X, self.coef_)

            if self.fit_intercept:
                y_pred += self.intercept_
            return y_pred
