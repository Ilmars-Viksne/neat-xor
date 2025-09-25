from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class EvolutionHistory:
    """Data class to store metrics from an evolution run."""
    generations: List[int] = field(default_factory=list)
    best_overall: List[float] = field(default_factory=list)
    gen_best: List[float] = field(default_factory=list)
    avg: List[float] = field(default_factory=list)
    species: List[int] = field(default_factory=list)