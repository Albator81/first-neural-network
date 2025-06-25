from typing import Callable, TypeVar, Generic
from numpy.typing import NDArray


T = TypeVar('T')


class Cost(Generic[T]):
    def __init__(self, f: Callable[[NDArray, NDArray], T],
                       df_dact_simplification: Callable[[NDArray, NDArray], NDArray],
                       name: str | None = None) -> None:
        """You'll need to know the activation function's derivative on the output layer"""
        self.f = f
        self.out_simp = df_dact_simplification
        if name is None:
            name = f.__name__
        self.name = name

    def get_cost(self, out: NDArray, target: NDArray) -> T:
        return self.f(out, target)

    def output_l_error(self, out: NDArray, target: NDArray) -> NDArray:
        return self.out_simp(out, target)

    def dcost_dweights(self, error_l: NDArray, a_lm1: NDArray) -> NDArray:
        """Calculate the derivative of the cost with respect to the weights.

        :param error_l: The error at layer l
        :param a_lm1: The activations of the previous layer (l-1)"""
        return error_l @ a_lm1.T

    def dcost_dbiases(self, error_l: NDArray) -> NDArray:
        """Calculate the derivative of the cost with respect to the biases.

        :param error_l: The error at layer l"""
        return error_l
