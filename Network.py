import numpy as np
from typing import Callable
from numpy.typing import NDArray

import json


def get(x, y):
    return x if x is not None else y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU(x):
    return np.maximum(x, 0)


def d_ReLU(x):
    return np.where(x > 0, 1, 0)


def softmax(z: NDArray) -> NDArray:
    # subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


class Network:
    activation = sigmoid
    derivation = d_sigmoid

    def __init__(self, lW: list[NDArray], lA: list[NDArray], lB: list[NDArray]) -> None:
        self.lW = lW
        self.lA = lA
        self.lB = lB

        self.L = len(lW) + 1 # (layer index begins at 0)
        self.lZ = [np.zeros_like(a) for a in lA]

        self.activations = [Network.activation]*(self.L-2) + [softmax]
        self.derivatives  = [Network.derivation]*(self.L-2) + [None]

    def set_input(self, *inputs):
        self.lA[0] = np.array(inputs, ndmin=2).T

    def layer_forward(self, l: int):
        W = self.lW[l]
        A = self.lA[l]
        B = self.lB[l]
        Z = W @ A + B

        self.lZ[l+1] = Z
        act = self.activations[l]
        self.lA[l+1] = act(Z)

    def forward(self):
        for l in range(self.L - 1):
            self.layer_forward(l)

    def cost(self, target: NDArray) -> float:
        """Cross-entropy for one-hot target + softmax outputs."""
        a = self.lA[self.L-1]
        # add tiny epsilon for numerical safety
        return -float(np.sum(target * np.log(a + 1e-8)))

    def error_output_layer(self, target: NDArray) -> NDArray:
        gradC = self.lA[self.L - 1] - target
        errorV = gradC
        return errorV

    def error_hidden_layer(self, l: int, errorVp1: NDArray) -> NDArray:
        """`errorVp1`: error vector of layer (`l`+1)"""
        W = self.lW[l]
        errorV = np.dot(W.T, errorVp1) * Network.derivation(self.lZ[l])
        return errorV
    
    def gradient_cost_biases(self, l: int, errorV: NDArray) -> NDArray:
        """`errorV`: error vector of layer `l`"""
        return errorV

    def gradient_cost_weights(self, l: int, errorV: NDArray) -> NDArray:
        """`errorV`: error vector of layer `l`"""
        A = self.lA[l]
        return np.dot(errorV, A.T)

    def backpropagate(self, target: NDArray, learning_rate: float):
        errorV = self.error_output_layer(target)

        for l in range(self.L - 2, -1, -1):
            gradW = self.gradient_cost_weights(l, errorV)
            gradB = self.gradient_cost_biases(l, errorV)

            self.lW[l] -= learning_rate * gradW
            self.lB[l] -= learning_rate * gradB

            if l > 0:
                errorV = self.error_hidden_layer(l, errorV)

    def train(self, input_output_pairs: list[tuple[tuple, tuple]], iterations: int, mode: str = "default"):
        def learn(target: NDArray, rate_func: Callable):
            cost = float(self.cost(target))
            learning_rate = rate_func(cost)

            self.backpropagate(target, learning_rate)


        # functions to tweak learning rate based on cost
        def constant(cost: float) -> float: return 0.01
        def default (cost: float) -> float: return 0.01  * (1 - np.exp(-cost))
        def crazy   (cost: float) -> float: return 0.08  * (1 - np.exp(-cost * 4))
        mode_to_func = {"constant": constant, "default": default, "crazy": crazy}
        rate_func = get(mode_to_func.get(mode), default)

        print(f"Training with mode: {rate_func.__name__}, Activation: {Network.activation.__name__}, Derivation: {Network.derivation.__name__}")
        print(f"Iterations: {iterations}, batch size: {len(input_output_pairs)}")

        sum_cost = 0.
        correct = 0
        for i in range(iterations):
            sum_cost = 0.
            correct = 0
            for input_, target in input_output_pairs:
                self.set_input(*input_)
                target = np.array(target, ndmin=2).T

                self.forward()
                if np.argmax(self.lA[self.L - 1]) == np.argmax(target): # dangerous if not only one max
                    correct += 1

                learn(target, rate_func)

                sum_cost += self.cost(target)
            if i % 100 == 0:
                width = len(str(iterations))
                print(f"Iteration {i:<{width}}, Average Cost: {sum_cost / len(input_output_pairs):.6f}, Accuracy: {correct / len(input_output_pairs):.2%}")
        
        print("\nTraining completed.\n")
        print(f"Final Cost: {sum_cost / len(input_output_pairs):.6f}")
        print(f"Final Accuracy: {correct / len(input_output_pairs):.2%}\n")


    def predict(self, *inputs: tuple, simplify=True):
        s = ""
        for input_ in inputs:
            self.set_input(*input_)
            self.forward()
            output = self.lA[self.L - 1].flatten().tolist()
            if simplify:
                input_ = [round(i, 3) for i in input_]
                output = [round(o, 3) for o in output]
            s += f"In: {input_} Out: {output}\n"
        return s

    def save(self, filename: str):
        data = {
            "Shape": [a.shape[0] for a in self.lA],
            "lW": [w.tolist() for w in self.lW],
            "lB": [b.tolist() for b in self.lB],
            "activations": [f.__name__ for f in self.activations],
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def from_json(cls, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
            lW = [np.array(w) for w in data["lW"]]
            lB = [np.array(b) for b in data["lB"]]
            lA = [np.zeros((shape, 1)) for shape in data["Shape"]]
        return cls(lW, lA, lB)
