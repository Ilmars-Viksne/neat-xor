from __future__ import annotations
from typing import List, Tuple, Callable, Iterable

from neat_ml import Problem

class XORProblem(Problem):
    """
    A simple problem to evolve a network that can solve the XOR logic gate.
    """
    def __init__(self, use_bias_as_input: bool = True):
        self.use_bias_as_input = use_bias_as_input

    def get_input_output_size(self) -> Tuple[int, int]:
        n_inputs = 3 if self.use_bias_as_input else 2
        return (n_inputs, 1)

    def _patterns(self) -> List[Tuple[List[float], List[float]]]:
        if self.use_bias_as_input:
            return [
                ([0.0, 0.0, 1.0], [0.0]),
                ([0.0, 1.0, 1.0], [1.0]),
                ([1.0, 0.0, 1.0], [1.0]),
                ([1.0, 1.0, 1.0], [0.0]),
            ]
        else:
            return [
                ([0.0, 0.0], [0.0]),
                ([0.0, 1.0], [1.0]),
                ([1.0, 0.0], [1.0]),
                ([1.0, 1.0], [0.0]),
            ]

    def evaluate(self, forward: Callable[[List[float]], List[float]], _genome) -> float:
        """
        The fitness is 4.0 minus the sum of squared errors over the four XOR patterns.
        A perfect score is 4.0.
        """
        sse = 0.0
        for x, y_true in self._patterns():
            y_pred = forward(x)
            y = max(0.0, min(1.0, y_pred[0]))
            sse += (y - y_true[0]) ** 2
        return 4.0 - sse

    def demo_samples(self) -> Iterable[Tuple[List[float], List[float]]]:
        return self._patterns()

    def goal_reached(self, forward: Callable[[List[float]], List[float]], _genome) -> bool:
        """The goal is reached if the network's output is within 0.01 of the target for all patterns."""
        tol = 1e-2
        for x, y_true in self._patterns():
            y_pred = forward(x)[0]
            if abs(y_pred - y_true[0]) > tol:
                return False
        return True