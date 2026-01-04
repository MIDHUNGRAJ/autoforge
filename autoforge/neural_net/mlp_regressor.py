from autoforge.base import BaseEstimator, require_fit
import numpy as np
from numpy.random import rand
from numpy.typing import NDArray
from autoforge.neural_net.activations import ACTIVATIONS


class MLPRegressor(BaseEstimator):
    """
    Multi-layer Perceptron Regressor

    This implementation supports multiple hidden layers and customizable
    activation functions for the hidden layers.

    Parameters
    ----------
    hidden_layer_size : tuple of int
        The number of neurons in each hidden layer.
        For example, (64, 32) creates two hidden layers with 64 and 32 neurons.

    learning_rate : float
        Learning rate for gradient descent weight updates.
        Controls how much the weights are adjusted during optimization.

    activation : str or BaseActivation
        Activation function used in the hidden layers.
        Supported strings: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'swish'.

    max_iter : int
        Maximum number of training iterations (epochs).

    random_state : int or None
        Random seed for reproducible weight initialization.
        If None, the random initialization is not reproducible.
    """

    def __init__(
        self,
        hidden_layer_size,
        learning_rate,
        activation,
        max_iter,
        random_state,
        batch_size,
    ) -> None:
        self.hidden_layer_size = hidden_layer_size
        self.lr = learning_rate
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size

    def _initialize_weight(self, n_features):
        """Initialize weights and biases for all network layers."""
        self.weight = []
        self.bias = []

        self.layer_size = [n_features] + list(self.hidden_layer_size) + [1]

        for i in range(len(self.layer_size) - 1):
            w = np.random.randn(self.layer_size[i], self.layer_size[i + 1]) * np.sqrt(
                2 / self.layer_size[i]
            )

            b = np.zeros((1, self.layer_size[i + 1]))

            self.weight.append(w)
            self.bias.append(b)

    def _forward_pass(self, X, predict=False):
        """Run a forward pass through the network."""
        A = X

        if not predict:
            self.Zs = []
            self.As = [X]

        for i in range(len(self.weight)):
            self.z = A @ self.weight[i] + self.bias[i]

            if not predict:
                self.Zs.append(self.z)

            if i == len(self.weight) - 1:
                self.y_hat = self.z
                A = self.z

            else:
                A = self.activation_func.forward(self.z)

            if not predict:
                self.As.append(A)

        return A

    def _backward_pass(self, X, y, y_pred):
        """Run backpropagation and update weights."""
        m = X.shape[0]

        dz = (y_pred - y) / m

        for i in reversed(range(len(self.weight))):
            dw = self.As[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)

            if i > 0:
                da = dz @ self.weight[i].T
                dz = da * self.activation_func.backward(self.Zs[i - 1])

            self.weight[i] -= self.lr * dw
            self.bias[i] -= self.lr * db

    def fit(self, X, y):
        """Train the neural network on input data."""
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.activation_func = ACTIVATIONS[self.activation]

        # Intialize weights
        self._initialize_weight(n_features)
        # Training loop
        for epoch in range(self.max_iter):
            idx = np.random.permutation(n_samples)

            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)

                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                y_hat = self._forward_pass(X_batch, predict=False)

                batch_loss = np.mean((y_batch - y_hat) ** 2)

                epoch_loss += batch_loss
                n_batches += 1

                self._backward_pass(X_batch, y_batch, y_hat)

            avg_loss = epoch_loss / n_batches

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {avg_loss:.4f}")

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        """Predict continuous target values."""
        X = np.asarray(X)

        A = self._forward_pass(X, predict=True)

        return A.flatten()

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        y_pred = self.predict(X)
        y = np.asarray(y)

        # Sum of squares of residuals
        ss_res = np.sum((y - y_pred.flatten()) ** 2)

        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # R² score
        r2_score = 1 - (ss_res / ss_tot)

        return r2_score
