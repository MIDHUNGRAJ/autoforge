import numpy as np
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from activations import Relu


class MLPRegressor:
    def __init__(
        self, hidden_layer_size, learning_rate, activation, max_iter, random_state
    ) -> None:
        self.hidden_layer_size = hidden_layer_size
        self.lr = learning_rate
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

    def backward(self, X, y, y_pred):
        m = X.shape[0]

        dz = (y_pred - y) / m

        for i in reversed(range(len(self.weight))):
            dw = self.As[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)

            if i > 0:
                da = dz @ self.weight[i].T
                dz = da * self.activation.backward(self.Zs[i - 1])

            self.weight[i] -= self.lr * dw
            self.bias[i] -= self.lr * db

    def fit(self, X, y):
        n_features = X.shape[1]

        self.activation = Relu()
        self.layer_size = [n_features] + list(self.hidden_layer_size) + [1]

        self.weight = []
        self.bias = []

        for i in range(len(self.layer_size) - 1):
            w = np.random.randn(self.layer_size[i], self.layer_size[i + 1]) * np.sqrt(
                2 / self.layer_size[i]
            )

            b = np.zeros((1, self.layer_size[i + 1]))

            self.weight.append(w)
            self.bias.append(b)

        for epoch in range(self.max_iter):
            A = X

            self.Zs = []
            self.As = [X]

            for i in range(len(self.weight)):
                self.z = A @ self.weight[i] + self.bias[i]
                self.Zs.append(self.z)

                if i == len(self.weight) - 1:
                    self.y_hat = self.z
                    A = self.z

                else:
                    A = self.activation.forward(self.z)

                self.As.append(A)

            loss = np.mean((y - self.y_hat) ** 2)

            self.backward(X, y, self.y_hat)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {loss:.4f}")

    def predict(self, X):
        A = X

        for i in range(len(self.weight)):
            Z = A @ self.weight[i] + self.bias[i]

            if i == len(self.weight) - 1:
                A = Z

            else:
                A = self.activation.forward(Z)

        return A.flatten()


X, y = make_regression(n_samples=1000, n_features=2, noise=10.0, random_state=42)

y = y.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = MLPRegressor(
    hidden_layer_size=(64,),
    learning_rate=0.01,
    activation="relu",
    max_iter=1000,
    random_state=42,
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

test_mse = np.mean((y_test_pred - y_test) ** 2)
print("Test MSE:", test_mse)

r2 = r2_score(y_test, y_test_pred)
print("R2 Score:", r2)
