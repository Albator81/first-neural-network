import numpy as np
from Network import Network
from network_setup import shape_network
import random

def create_input_output_pairs(n: int) -> list[tuple[tuple, tuple]]:
    """
    Input: 3
    Output: 3 (confidence that number is the biggest among the three)
    """

    pairs = []
    for _ in range(1000):  # Generate 1000 random pairs
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
    net = Network(*shape_network(3, 5, 5, 5, 3))
    # net = Network.from_json("network.json")

    pairs = create_input_output_pairs(50)

    net.train(pairs, iterations=300, mode="constant")
    net.save("network.json")

    print("Some Tests:")
    print(net.predict(*[(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1) for _ in range(10)]))

if __name__ == "__main__":
    main()
