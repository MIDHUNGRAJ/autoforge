import numpy as np
from autoforge.base import BaseEstimator, require_fit
from activations import sigmoid
from activations import Relu


class MLPClassifier(BaseEstimator):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
    """

    def __init__(
        self, hidden_layer_sizes, activation, learning_rate, max_iter, random_state
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = learning_rate
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
        y = y.reshape(-1, 1)

        self.activation = Relu()

        n_features = X.shape[1]

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

        # Training Loop
        for epoch in range(self.max_iter):
            A = X
            self.Zs = []
            self.As = [X]

            for i in range(len(self.weight)):
                self.z = A @ self.weight[i] + self.bias[i]
                self.Zs.append(self.z)

                if i == len(self.weight) - 1:
                    A = sigmoid.forward(self.z)

                else:
                    A = self.activation.forward(self.z)

                self.As.append(A)

            y_hat = A

            loss = -np.mean(
                y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8)
            )
            self.backward(X, y, y_hat)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {loss:.4f}")

    @require_fit
    def predict(self, X):
        X = np.asarray(X)

        A = X

        for i in range(len(self.weight)):
            Z = A @ self.weight[i] + self.bias[i]

            if i == len(self.weight) - 1:
                A = sigmoid.forward(Z)

            else:
                A = self.activation.forward(Z)

        return (A > 0.5).astype(int).flatten()


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    learning_rate=0.01,
    max_iter=1000,
    random_state=42,
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# End
