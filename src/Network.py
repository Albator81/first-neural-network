import numpy as np
from typing import Callable, Sequence, Any
from numpy.typing import NDArray
import random
import json

from src.activation.functions import sigmoid_act, relu_act, softmax_act
from src.cost.functions import cross_entropy_softmax_cost
from src.cost.Cost import Cost


def get(x, y):
    return x if x is not None else y


class LR_Schedule:
    schedules: dict[str, 'LR_Schedule'] = {}
    def __init__(self, func: Callable[[float], float], name: str | None = None):
        self.func = func

        if name is None:
            name = func.__name__
        self.name = name
        if name in LR_Schedule.schedules:
            raise ValueError(f"Learning rate schedule with name '{name}' already exists.")
        LR_Schedule.schedules[name] = self

    def __call__(self, cost: float) -> float:
        return self.func(cost)

constant_lr = LR_Schedule(lambda c: 0.01, "constant")
default_lr  = LR_Schedule(lambda c: 0.03*(1-np.exp(-c)), "default")
high_lr    = LR_Schedule(lambda c: 0.08*(1-np.exp(-4*c)), "high")


class Network:

    def __init__(
            self, 
            lW: list[NDArray], 
            lA: list[NDArray], 
            lB: list[NDArray], 
            activations: Sequence | None = None, 
            cost:        Cost     | None = None
        ) -> None:

        self.lW = lW
        self.lA = lA
        self.lB = lB

        self.L = len(lW) + 1 # (layer index begins at 0)
        self.lZ = [np.zeros_like(a) for a in lA]

        if activations is None:
            activations = [sigmoid_act]*(self.L-2) + [softmax_act]
        self.activations = activations

        if cost is None:
            cost = cross_entropy_softmax_cost
        self.cost = cost

    def set_input(self, *inputs):
        self.lA[0] = np.array(inputs, ndmin=2).T

    def _layer_forward(self, l: int):
        W = self.lW[l]
        A = self.lA[l]
        B = self.lB[l]
        Z = W @ A + B

        self.lZ[l+1] = Z
        act = self.activations[l].f
        self.lA[l+1] = act(Z)

    def forward(self):
        for l in range(self.L - 1):
            self._layer_forward(l)

    def error_output_layer(self, target: NDArray) -> NDArray:
        return self.cost.output_l_error(self.lA[self.L - 1], target)

    def error_hidden_layer(self, l: int, errorV_p1: NDArray) -> NDArray:
        """`error_p1`: error vector of layer (`l`+1)"""
        W = self.lW[l]
        errorV = np.dot(W.T, errorV_p1)
        if (df := self.activations[l].df) is not None:
            errorV *= df(self.lZ[l])

        return errorV

    def gradient_cost_biases(self, errorV_l: NDArray) -> NDArray:
        """`errorV_l`: error vector of layer `l`"""
        return self.cost.dcost_dbiases(errorV_l)

    def gradient_cost_weights(self, l: int, errorV: NDArray) -> NDArray:
        """`errorV`: error vector of layer `l`"""
        return self.cost.dcost_dweights(errorV, self.lA[l])

    def backpropagate(self, target: NDArray, learning_rate: float):
        errorV = self.error_output_layer(target)

        for l in range(self.L - 2, -1, -1):
            gradW = self.gradient_cost_weights(l, errorV)
            gradB = self.gradient_cost_biases(errorV)

            self.lW[l] -= learning_rate * gradW
            self.lB[l] -= learning_rate * gradB

            if l > 0:
                errorV = self.error_hidden_layer(l, errorV)

    def get_cost(self, target: NDArray):
        return self.cost.get_cost(self.lA[self.L - 1], target)

    def train(self, input_output_pairs: list[tuple[tuple, tuple]], iterations: int, mode: str = "default"):
        rate_func = LR_Schedule.schedules.get(mode, constant_lr)

        print(f"Training with Mode: {rate_func.name}")
        print(f"Iterations: {iterations}, Batch Size: {len(input_output_pairs)}")

        batch_size = len(input_output_pairs)
        width = len(str(iterations))

        sum_cost = 0.0
        correct = 0
        for i in range(iterations):
            sum_cost = 0.0
            correct = 0

            # random.shuffle(input_output_pairs)

            for input_, target in input_output_pairs:
                self.set_input(*input_)
                target = np.array(target, ndmin=2).T

                self.forward()
                output = self.lA[self.L - 1]

                if np.argmax(output) == np.argmax(target):
                    correct += 1

                cost_value = self.get_cost(target)
                sum_cost += cost_value
                self.backpropagate(target, rate_func(cost_value))

            if i % 100 == 0:
                avg_cost = sum_cost / batch_size
                accuracy = correct / batch_size
                print(f"Iteration {i:<{width}}, Average Cost: {avg_cost:.6f}, Accuracy: {accuracy:.2%}")

        final_cost = sum_cost / batch_size
        final_accuracy = correct / batch_size

        print("\nTraining completed.\n")
        print(f"Final Cost: {final_cost:.6f}")
        print(f"Final Accuracy: {final_accuracy:.2%}\n")

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
            "shape": [a.shape[0] for a in self.lA],
            "lW": [w.tolist() for w in self.lW],
            "lB": [b.tolist() for b in self.lB],
        }
        data['cost'] = self.cost.name
        data['activations'] = [act.name for act in self.activations]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def from_json(cls, filename: str):
        """Get the Weights, Biases and Activations values from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            lW = [np.array(w) for w in data["lW"]]
            lB = [np.array(b) for b in data["lB"]]
            lA = [np.zeros((shape, 1)) for shape in data["shape"]]

        return cls(lW, lA, lB)
