import numpy as np
from src.Network import Network
from network_setup import shape_network
import random

def create_input_output_pairs(n: int) -> list[tuple[tuple, tuple]]:
    """
    Input: 3
    Output: 3 (confidence that number is the biggest among the three)
    """

    pairs = []
    for _ in range(n):
        numbers = [random.random()*2-1 for _ in range(3)]
        a = [tuple(numbers)]

        max_index = numbers.index(max(numbers))
        output = [0., 0., 0.]
        output[max_index] = 1.0
        a.append(tuple(output))
        pairs.append(tuple(a))
    return pairs


def main():
    """
    Idea: 
    Input:  3 (numbers)
    Hidden: 
    Output: 3 (confidence that number is the biggest among the three)
    """
    # net = Network(*shape_network(3, 3, 3))
    net = Network.from_json("networks/network3.json")

    pairs = create_input_output_pairs(10_000)

    net.train(pairs, batch_size=200, epochs=5000, mode="default")
    net.save("networks/network3.json")

    # print("Some Tests:")
    # print(net.predict(*[(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1) for _ in range(10)]))
    # print("Precise Tests:")
    # print(net.predict((0.9, 0.91, 0.92), (-0.99, -0.98, -0.97), (0.0001, 0.0002, 0.00021), (0., -1.0, 1.0), simplify=False))

    # print("\nTesting accuracy:")
    # pairs = create_input_output_pairs(10_000)
    # print(net.test_accuracy(pairs))

if __name__ == "__main__":
    main()
