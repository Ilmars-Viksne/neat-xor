"""
A Python implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
This library provides the tools to evolve neural networks for a variety of problems.
"""
from .genome import Genome, NodeGene, ConnectionGene, NodeType
from .problem import Problem
from .persistence import save_genome_json, load_genome_json
from .neat import (
    NEATConfig,
    run_neat,
    decode_to_network,
    ACTIVATION_FUNCTIONS
)
from .history import EvolutionHistory
from .visualization import visualize_genome, plot_history

__all__ = [
    "Genome",
    "NodeGene",
    "ConnectionGene",
    "NodeType",
    "Problem",
    "save_genome_json",
    "load_genome_json",
    "NEATConfig",
    "run_neat",
    "decode_to_network",
    "EvolutionHistory",
    "visualize_genome",
    "plot_history",
    "ACTIVATION_FUNCTIONS",
]