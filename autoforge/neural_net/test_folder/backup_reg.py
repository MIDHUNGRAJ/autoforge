import numpy as np


class MLPRegressor:
    def __init__(self, d_in=2, d_h=32):
        self.w1 = np.random.randn(d_in, d_h) * 0.01
        self.b1 = np.zeros((1, d_h))
        self.w2 = np.random.randn(d_h, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def act_fun(self, x):  # Activation function
        return np.maximum(0, x)

    def act_fun_grad(self, x):  # Derivation of activation function
        return (x > 0).astype(float)

    def forward(self, X):  # Forward propagaiton
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.act_fun(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred, lr):
        m = X.shape[0]

        dz2 = (y_pred - y) / m

        dw2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * self.act_fun_grad(self.z1)

        dw1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1


from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=2, noise=10.0, random_state=42)

y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = MLPRegressor(d_in=2, d_h=32)

epochs = 2000
lr = 0.01

for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train)

    # MSE loss
    loss = np.mean((y_pred - y_train) ** 2)

    # Backward pass
    model.backward(X_train, y_train, y_pred, lr)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

y_test_pred = model.forward(X_test)
print(y_test.shape)

test_mse = np.mean((y_test_pred - y_test) ** 2)
print("Test MSE:", test_mse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_test_pred)
print("R2 Score:", r2)
