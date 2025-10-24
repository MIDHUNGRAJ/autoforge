import numpy as np
from autoforge.base import BaseEstimator, require_fit


class LinearRegression(BaseEstimator):
    """
    Ordinary Least Squares Linear Regression.

    This implementation solves the normal equations:
        theta = (X^T X)^{-1} X^T y

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate and include an intercept term in the model.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term (intercept) in the linear model. Set to 0.0 if fit_intercept=False.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit linear regression model to training data using the normal equation.

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
        if self.fit_intercept:
            # Add a column of ones for the intercept term
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Solve normal equation for theta
        # theta = (X^T X)^(-1) X^T y
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)

        if self.fit_intercept:
            self.intercept_ = self.theta[0]
            self.coef_ = self.theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        """
        Predict target values using the linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples for which to predict target values.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.fit_intercept:
            # Add a column of ones for the intercept term
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute predictions
        predict = X @ self.theta

        return predict
