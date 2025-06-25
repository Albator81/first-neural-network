from typing import Callable
from numpy.typing import NDArray


class Activation:
    def __init__(self, f: Callable[[NDArray], NDArray], df: Callable[[NDArray], NDArray] | None, 
                 name: str | None = None) -> None:
        self.f  = f
        self.df = df
        self.name = name if name is not None else (f.__name__ + '_act')
