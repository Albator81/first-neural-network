import numpy as np
from numpy.typing import NDArray
from src.cost.Cost import Cost


def cross_entropy(output: NDArray, target: NDArray) -> float:
    """Cross-entropy for one-hot target + softmax outputs.
    
    :param output: The output layer
    
    :return: The cross-entropy cost."""
    # add tiny epsilon for numerical safety
    return -float(np.sum(target * np.log(output + 1e-8)))

def dcross_entropy_softmax_simplification(out: NDArray, target: NDArray) -> NDArray:
    """:param out: The output layer."""
    return out - target


cross_entropy_softmax_cost = Cost(cross_entropy, dcross_entropy_softmax_simplification, 'cross_entropy_softmax_cost')
