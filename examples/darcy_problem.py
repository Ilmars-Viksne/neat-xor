"""
Defines the Darcy friction factor problem for the neat_ml library.

This module provides the `DarcyProblem` class, which encapsulates the dataset
and evaluation logic for predicting the Darcy friction factor using NEAT.
"""

import numpy as np
from typing import List, Tuple, Callable
from neat_ml.problem import Problem

# --- Darcy Friction Factor Calculation (Helper Functions) ---

def _f_haaland(Re, eps_over_D):
    Re = np.asarray(Re, dtype=float)
    Re = np.clip(Re, 1e-12, None)
    term = ((eps_over_D / 3.7) ** 1.11) + 6.9 / Re
    return (-1.8 * np.log10(term)) ** -2

def _f_colebrook(Re, eps_over_D, tol=1e-8, max_iter=60):
    Re = np.asarray(Re, dtype=float)
    f = np.zeros_like(Re)
    laminar_mask = Re < 2300.0
    if np.any(laminar_mask):
        f[laminar_mask] = 64.0 / np.clip(Re[laminar_mask], 1e-12, None)
    turbulent_mask = ~laminar_mask
    if np.any(turbulent_mask):
        Re_t = Re[turbulent_mask]
        eps_D_t = eps_over_D if np.isscalar(eps_over_D) else eps_over_D[turbulent_mask]
        f_t = _f_haaland(Re_t, eps_D_t)
        eps_term = eps_D_t / 3.7
        for _ in range(max_iter):
            f_sqrt = np.sqrt(f_t)
            lhs = -2.0 * np.log10(eps_term + 2.51 / (Re_t * f_sqrt))
            f_new = 1.0 / (lhs ** 2)
            if np.all(np.abs(f_new - f_t) <= tol * np.maximum(f_t, 1e-16)):
                f_t = f_new
                break
            f_t = f_new
        f[turbulent_mask] = f_t
    return f

# --- Problem Definition ---

class DarcyProblem(Problem):
    """
    Implements the Darcy friction factor prediction problem for NEAT.
    """
    def __init__(self):
        # Generate training data upon initialization
        self.inputs, self.outputs = self._generate_data()

    def _generate_data(self):
        """Generates a grid of training points."""
        re_log_min, re_log_max = 2.5, 8.0
        eps_min, eps_max = 0.0, 0.005
        n_re, n_eps = 20, 100

        re_log_values = np.linspace(re_log_min, re_log_max, n_re)
        eps_values = np.linspace(eps_min, eps_max, n_eps)
        re_values = 10 ** re_log_values

        inputs, outputs = [], []
        for i, re_log in enumerate(re_log_values):
            for j, eps in enumerate(eps_values):
                re = re_values[i]
                inputs.append([re_log, eps, 1.0]) # Use list of floats
                f = _f_colebrook(np.array([re]), np.array([eps]))
                outputs.append(list(f)) # Use list of floats
        return inputs, outputs

    def get_input_output_size(self) -> Tuple[int, int]:
        """Returns the number of inputs (3) and outputs (1)."""
        return 3, 1

    def evaluate(self, forward: Callable[[List[float]], List[float]], genome) -> float:
        """
        Evaluates a genome's fitness based on the inverse of the mean squared error (MSE).
        """
        sse = 0.0
        for xi, xo in zip(self.inputs, self.outputs):
            output = forward(xi)
            sse += (output[0] - xo[0]) ** 2

        mse = sse / len(self.inputs)
        # Add a small epsilon to avoid division by zero for a perfect score
        fitness = 1.0 / (1.0 + mse)
        return fitness

    def demo_samples(self):
        """Provides the training data as demonstration samples."""
        return zip(self.inputs, self.outputs)

# Expose the Colebrook function at the module level for the run script to use
f_colebrook = _f_colebrook