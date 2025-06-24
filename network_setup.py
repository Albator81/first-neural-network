import numpy as np


# def shape_network(*layers: int):
#     lN = [np.zeros((n, 1)) for n in layers]
#     lW = [np.zeros((layers[i+1], layers[i])) for i in range(len(layers)-1)]
#     lB = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]
#     # dtype=float64
#     return lW, lN, lB


def shape_network(*layers: int):
    lN = [np.zeros((n, 1)) for n in layers]
    lW = [np.random.uniform(0, 1, (layers[i+1], layers[i])) for i in range(len(layers)-1)]
    lB = [np.random.uniform(0, 1, (layers[i], 1)) for i in range(1, len(layers))]
    return lW, lN, lB

