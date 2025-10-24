import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true: array-like of shape (n_samples,)
    - y_pred: array-like of shape (n_samples,)

    Returns:
    - float: Mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_true: array-like of shape (n_samples,)
    - y_pred: array-like of shape (n_samples,)

    Returns:
    - float: Root mean squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

