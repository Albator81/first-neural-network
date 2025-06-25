import numpy as np
from numpy.typing import NDArray


def array_to_str(arr: NDArray, deci: int = 3, sep: str = ' ') -> str:
    return sep.join([f"{x:.{deci}f}" for x in arr.flatten().tolist()])


class EpochStatistics:
    def __init__(self, *, array_element_sep: str = ' ', decimals: int = 3) -> None:
        self.sep = array_element_sep
        self.deci = decimals
        self.reset()

    def best_cost_str(self) -> str:
        return f"{self.best_cost:.{self.deci}f}" if self.best_cost is not None else "None"
    def worst_cost_str(self) -> str:
        return f"{self.worst_cost:.{self.deci}f}" if self.worst_cost is not None else "None"
    def worst_out_str(self) -> str:
        return self.worst_out if self.worst_out is not None else "None"
    def best_out_str(self) -> str:
        return self.best_out if self.best_out is not None else "None"
    def avg_cost_str(self) -> str:
        return self.avg_cost if self.avg_cost is not None else "None"
    def n_correct_str(self) -> str:
        return f"{self.n_correct:>{len(str(self.batch_i))}}/{self.batch_i}" if self.n_correct is not None else "None"
    def n_incorrect_str(self) -> str:
        return f"{self.n_incorrect:<{len(str(self.batch_i))}}/{self.batch_i}" if self.n_incorrect is not None else "None"
    def accuracy_str(self) -> str:
        return self.accuracy if self.accuracy is not None else "None"
    def l_incorrect_in_out_str(self, sep: str = '\n') -> str:
        return sep.join([' -> '.join(pair) for pair in self.l_incorrect_in_out]) if len(self.l_incorrect_in_out) != 0 else "None"
    def current_lr_str(self) -> str:
        return str(self.current_lr) if self.current_lr is not None else "None"
    def epoch_str(self) -> str:
        return str(self.epoch)

    def best_cost_labeled(self) -> str:
        return 'Best Cost: ' + self.best_cost_str()
    def worst_cost_labeled(self) -> str:
        return 'Worst Cost: ' + self.worst_cost_str()
    def worst_out_labeled(self) -> str:
        return 'Worst Out: ' + self.worst_out_str()
    def best_out_labeled(self) -> str:
        return 'Best Out: ' + self.best_out_str()
    def avg_cost_labeled(self) -> str:
        return 'Avg Cost: ' + self.avg_cost_str()
    def n_correct_labeled(self) -> str:
        return 'Correct: ' + self.n_correct_str()
    def n_incorrect_labeled(self) -> str:
        return 'Incorrect: ' + self.n_incorrect_str()
    def accuracy_labeled(self) -> str:
        return 'Accuracy: ' + self.accuracy_str()
    def l_incorrect_in_out_labeled(self, sep: str = '\n') -> str:
        return 'Incorrect Cases: ' + self.l_incorrect_in_out_str(sep)
    def current_lr_labeled(self) -> str:
        return 'Current LearningRate: ' + self.current_lr_str()
    def epoch_labeled(self) -> str:
        return 'Epoch: ' + self.epoch_str()

    def new_epoch(self):
        self.batch_i = 0

        self.worst_out = None
        self.best_out = None

        self.worst_cost = None
        self.best_cost = None

        self.sum_cost = 0.
        self.avg_cost = None

        self.n_correct = 0
        self.n_incorrect = 0
        self.accuracy = None

        self.l_incorrect_in_out = []

        self.current_lr = None

        self.epoch += 1
    
    def reset(self):
        self.epoch = 0 # ...
        self.new_epoch()
        self.epoch = 0 # yes

    def update(self, input_: NDArray, output: NDArray, target: NDArray, is_correct: bool, cost, learnrate):
        """Call this method after every update to the neural network"""
        self.batch_i += 1

        self.current_lr = learnrate

        self.sum_cost += cost
        self.avg_cost = f"{self.sum_cost / self.batch_i:.{self.deci}f}"

        if is_correct:
            self.n_correct += 1
        else:
            self.n_incorrect += 1
            self.l_incorrect_in_out.append((array_to_str(input_, self.deci, self.sep), array_to_str(output, self.deci, self.sep)))

        self.accuracy = f"{self.n_correct / self.batch_i:.2%}"
        self.accuracy = f"{self.accuracy:>7}"


        if self.worst_cost is None or self.worst_cost < cost:
            self.worst_cost = cost
            self.worst_out = array_to_str(output, self.deci, self.sep)
        
        if self.best_cost is None or self.best_cost > cost:
            self.best_cost = cost
            self.best_out = array_to_str(output, self.deci, self.sep)
