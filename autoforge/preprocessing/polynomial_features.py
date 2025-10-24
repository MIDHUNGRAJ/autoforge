import numpy as np
from autoforge.base import BaseEstimator


class PolynomialFeatures(BaseEstimator):
    """
    Generate polynomial features for a single input feature.

    This transformer generates a new feature matrix consisting of all powers
    of the single input feature up to the specified degree.

    For example, for input feature x and degree=3:
        output = [1, x, x^2, x^3]   if include_bias=True
        output = [x, x^2, x^3]     if include_bias=False

    Parameters
    ----------
    degree : int, default=2
        The highest polynomial degree to generate.

    include_bias : bool, default=True
        If True, include a bias (constant) feature equal to 1.

    Attributes
    ----------
    n_output_features_ : int
        The number of output features produced.

    Notes
    -----
    - This implementation only handles a single input feature (one column).
      For multi-feature polynomial expansions, additional logic is needed.
    """

    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit_transform(self, X):
        """
        Fit to data, then transform it to polynomial features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 1)
            The data to transform. Must be a single feature.

        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Transformed feature matrix including polynomial features.
        """
        n_samples = X.shape[0]

        # Determine number of output columns
        n_cols = self.degree + 1 if self.include_bias else self.degree
        X_poly = np.ones((n_samples, n_cols))

        col_idx = 0
        if self.include_bias:
            # First column is ones for the intercept
            col_idx += 1

        for d in range(1, self.degree + 1):
            # Raise the single feature to the power d
            X_poly[:, col_idx] = X[:, 0] ** d
            col_idx += 1

        self.n_output_features_ = n_cols
        return X_poly
