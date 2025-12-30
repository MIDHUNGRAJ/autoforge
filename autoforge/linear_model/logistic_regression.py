import numpy as np
from autoforge.base import BaseEstimator, require_fit
from autoforge.utils import sigmoid


class LogisticRegression(BaseEstimator):
    """
    A simple implementation of logistic regression using batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size used during gradient descent updates.

    fit_intercept : bool, default=True
        Whether to include an intercept (bias term) in the model.

    max_iter : int, default=100
        Number of iterations over the training data for optimization.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_features + 1,)
        The learned coefficients of the model. Includes the intercept if fit_intercept=True.
    """

    def __init__(self, learning_rate=0.01, fit_intercept=True, max_iter=100):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def _add_bias(self, X):
        """
        Add a bias (intercept) column of ones to the input matrix X if fit_intercept=True.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        X_new : ndarray
            Input matrix with an added bias column if applicable.
        """
        if self.fit_intercept:
            # Add a column of ones as the first column for the intercept term
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X, y):
        """
        Fit the logistic regression model to training data using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Binary target vector (values 0 or 1).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._add_bias(X)
        n_samples, n_features = X.shape

        # Initialize weights to zero
        self.coef_ = np.zeros(n_features)

        for _ in range(self.max_iter):
            # Compute linear combination
            z = X @ self.coef_
            # Apply sigmoid activation for probability estimates
            pred = sigmoid(z)
            # Compute gradient of the loss w.r.t. coefficients
            gradient = (X.T @ (pred - y)) / n_samples
            # Update coefficients in the opposite direction of the gradient
            self.coef_ -= self.learning_rate * gradient

        # Mark the model as fitted
        self._mark_fitted()

        return self

    @require_fit
    def predict_proba(self, X):
        """
        Compute predicted probabilities for input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix of samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Predicted probabilities of the positive class.
        """
        X = self._add_bias(X)
        return sigmoid(X @ self.coef_)

    @require_fit
    def predict(self, X):
        """
        Predict binary class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix of samples to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        # Predict probabilities and convert to binary output using 0.5 threshold
        return (self.predict_proba(X) >= 0.5).astype(int)
