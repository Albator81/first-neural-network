import numpy as np


def shape_network(*layers):
    # layers = [n_input, n_h1, …, n_output]
    lA = [np.zeros((n,1)) for n in layers]
    lW = []
    lB = []
    for inp, out in zip(layers, layers[1:]):
        # Xavier: std = sqrt(1 / inp)
        limit = np.sqrt(1.0 / inp)
        W = np.random.uniform(-limit, limit, (out, inp))
        b = np.zeros((out,1))
        lW.append(W)
        lB.append(b)
    return lW, lA, lB
