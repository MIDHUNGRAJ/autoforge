import numpy as np


# Check function for forward and backward
def check_forward_backward(actfun):
    if not hasattr(actfun, "forward") or not callable(actfun.forward):
        raise TypeError(f"Class '{actfun.__name__}' is missing a 'forward' ")

    if not hasattr(actfun, "backward") or not callable(actfun.backward):
        raise TypeError(f"Class '{actfun.__name__}' is missing a 'backward' ")

    return actfun


@check_forward_backward
class Relu:
    @staticmethod
    def forward(X):
        return np.maximum(0, X)

    @staticmethod
    def backward(X):
        return (X > 0).astype(float)


@check_forward_backward
class Sigmoid:
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, X):
        A = self.forward(X)
        return A * (1 - A)


# Legacy support - for backwards compatibility
class sigmoid:
    @staticmethod
    def forward(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def backward(X):
        A = sigmoid.forward(X)
        return A


ACTIVATIONS = {
    "relu": Relu,
    "sigmoid": Sigmoid,
}
