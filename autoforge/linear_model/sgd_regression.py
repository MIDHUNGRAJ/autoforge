import numpy as np
from autoforge.base import BaseEstimator, require_fit


class SGDRegressor(BaseEstimator):
    """
    Linear regression model optimized via Stochastic Gradient Descent (SGD).

    This implementation updates model parameters incrementally for each sample,
    rather than solving the normal equations or using batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.

    fit_intercept : bool, default=True
        Whether to calculate and include an intercept term in the model.

    max_iter : int, default=1000
        Total number of passes (epochs) over the training dataset.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the linear model.

    intercept_ : float
        Intercept (bias) term. None if fit_intercept=False.
    """

    def __init__(self, learning_rate, fit_intercept=True, max_iter=1000):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the SGDRegressor to training data.

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

                # Update coefficients
                self.coef_ -= self.learning_rate * error * xi

                # Update intercept if applicable
                if self.fit_intercept:
                    self.intercept_ -= self.learning_rate * error

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        """
        Predict target values using the trained SGDRegressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Compute predictions as dot product of X and coefficients
        y_pred = np.dot(X, self.coef_)

        if self.fit_intercept:
            y_pred += self.intercept_

        return y_pred
