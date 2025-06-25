import numpy as np
from numpy.typing import NDArray
from src.activation.Activation import Activation


def sigmoid  (z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))
def d_sigmoid(z: NDArray) -> NDArray:
    s = sigmoid(z)
    return s * (1 - s)


def relu  (z: NDArray) -> NDArray:
    return np.maximum(z, 0)
def d_relu(z: NDArray) -> NDArray:
    return np.where(z > 0, 1, 0)


def softmax(z: NDArray) -> NDArray:
    # subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


sigmoid_act = Activation(sigmoid, d_sigmoid)
relu_act    = Activation(relu, d_relu)
softmax_act = Activation(softmax, None)
