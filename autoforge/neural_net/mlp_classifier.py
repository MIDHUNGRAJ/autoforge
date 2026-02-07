import numpy as np

from autoforge.base import BaseEstimator, require_fit
from autoforge.neural_net.activations import ACTIVATIONS, sigmoid


class MLPClassifier(BaseEstimator):
    """
    Multi-layer Perceptron Classifier

    A feedforward neural network for binary classification using backpropagation
    with binary cross-entropy loss and sigmoid output activation.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        The number of neurons in each hidden layer.
        For example, (64, 32) creates two hidden layers with 64 and 32 neurons.
    activation : str or BaseActivation, default='relu'
        Activation function for the hidden layers.
        Supported strings: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'swish'.
    learning_rate : float, default=0.01
        Learning rate for gradient descent weight updates.
        Controls the step size during optimization.
    max_iter : int, default=200
        Maximum number of training iterations (epochs).
    random_state : int or None, default=None
        Random seed for reproducible weight initialization.
        Pass an integer for reproducible output across multiple function calls.

    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        learning_rate=0.01,
        max_iter=200,
        random_state=None,
        batch_size=32,
        solver="adam",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.solver = solver

    def _initialize_weight(self, n_features):
        """Initialize network weights"""
        self.weight = []
        self.bias = []

        self.layer_size = [n_features] + list(self.hidden_layer_sizes) + [1]

        for i in range(len(self.layer_size) - 1):
            w = np.random.randn(self.layer_size[i], self.layer_size[i + 1]) * np.sqrt(
                2 / self.layer_size[i]
            )

            b = np.zeros((1, self.layer_size[i + 1]))

            self.weight.append(w)
            self.bias.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weight]
        self.v_w = [np.zeros_like(w) for w in self.weight]
        self.m_b = [np.zeros_like(b) for b in self.bias]
        self.v_b = [np.zeros_like(b) for b in self.bias]
        self.t = 0

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _forward_pass(self, X, predict=False):
        """Perform forward propagation through the network."""
        A = X

        if not predict:
            self.Zs = []
            self.As = [X]

        for i in range(len(self.weight)):
            # Compute pre-activation
            self.z = A @ self.weight[i] + self.bias[i]

            if not predict:
                self.Zs.append(self.z)

            # Apply activation function
            if i == len(self.weight) - 1:
                # Output layer: sigmoid for binary classification
                A = sigmoid.forward(self.z)

            else:
                # Hidden layers: use specified activation
                A = self.activation_func.forward(self.z)

            if not predict:
                self.As.append(A)

        return A

    def _backward_pass_sgd(self, X, y, y_pred):
        """Perform backpropagation to compute gradients and update weights."""
        m = X.shape[0]

        # Compute output gradient
        dz = (y_pred - y) / m

        # Backpropagate through layers
        for i in reversed(range(len(self.weight))):
            # Compute gradient for current layer
            dw = self.As[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)

            # Propagate gradient to previous layer
            if i > 0:
                da = dz @ self.weight[i].T
                dz = da * self.activation_func.backward(self.Zs[i - 1])

            # Update weights using optimizer
            self.weight[i] -= self.lr * dw
            self.bias[i] -= self.lr * db

    def _backward_pass_adam(self, X, y, y_pred):
        """Run backpropagation and update weights."""
        m = X.shape[0]

        dz = (y_pred - y) / m

        grad_w = []
        grad_b = []

        self.t += 1

        for i in reversed(range(len(self.weight))):
            dw = self.As[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)

            grad_w.insert(0, dw)
            grad_b.insert(0, db)

            if i > 0:
                da = dz @ self.weight[i].T
                dz = da * self.activation_func.backward(self.Zs[i - 1])

            # self.weight[i] -= self.lr * dw
            # self.bias[i] -= self.lr * db

        for i in range(len(self.weight)):
            dw = grad_w[i]
            db = grad_b[i]

            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * db**2

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * db**2

            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_b[i] / (1 - self.beta2**self.t)

            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            self.weight[i] -= (self.lr * m_w_hat) / (np.sqrt(v_w_hat) + self.eps)
            self.bias[i] -= (self.lr * m_b_hat) / (np.sqrt(v_b_hat) + self.eps)

    def fit(self, X, y):
        """Fit the MLP classifier to training data."""
        y = y.reshape(-1, 1)

        self.activation_func = ACTIVATIONS[self.activation]
        # print(_activ)

        # Initialize components
        n_samples, n_features = X.shape

        # Intiaialize weights
        self._initialize_weight(n_features)
        # Training Loop
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

                batch_loss = -np.mean(
                    y_batch * np.log(y_hat + 1e-8)
                    + (1 - y_batch) * np.log(1 - y_hat + 1e-8)
                )

                if self.solver == "adam":
                    self._backward_pass_adam(X_batch, y_batch, y_hat)
                elif self.solver == "sgd":
                    self._backward_pass_sgd(X_batch, y_batch, y_hat)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {avg_loss:.4f}")

        self._mark_fitted()
        return self

    @require_fit
    def predict(self, X):
        """Predict binary class labels for samples."""
        X = np.asarray(X)

        return (self._predict_proba(X) > 0.5).astype(int).flatten()

    def _predict_proba(self, X):
        """Predict class probabilities for samples."""
        # Forward pass without storing activations (memory efficient)
        A = self._forward_pass(X, predict=True)
        return A

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# End
