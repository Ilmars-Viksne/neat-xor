from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Iterable

class Problem(ABC):
    """
    Abstract base class for defining a problem for NEAT to solve.
    A problem must specify its input/output dimensions and provide an
    evaluation function.
    """
    @abstractmethod
    def get_input_output_size(self) -> Tuple[int, int]:
        """
        Returns the number of input and output nodes required for this problem.
        Example: (2, 1) for a 2-input, 1-output network.
        """
        ...

    @abstractmethod
    def evaluate(self, forward: Callable[[List[float]], List[float]], genome) -> float:
        """
        Evaluates a genome's fitness.
        The `forward` function is the phenotype (neural network) of the genome.
        It takes a list of floats (inputs) and returns a list of floats (outputs).
        The `genome` object is also passed in case its structure is needed for evaluation.
        Should return a single float representing the fitness score.
        """
        ...

    def demo_samples(self) -> Iterable[Tuple[List[float], List[float]]]:
        """
        Optional: Provides a set of input/output pairs for demonstrating
        the champion's performance after training.
        """
        return []

    def goal_reached(self, forward: Callable[[List[float]], List[float]], genome) -> bool:
        """
        Optional: Determines if the problem's goal has been met,
        allowing for early stopping of the evolution.
        By default, this is never reached.
        """
        return False