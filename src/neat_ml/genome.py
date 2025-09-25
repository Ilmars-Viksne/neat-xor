from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List

# =========================
# Gene & Genome Structures
# =========================
class NodeType:
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    BIAS = "bias"

@dataclass
class NodeGene:
    id: int
    type: str
    activation: str = 'tanh'

@dataclass
class ConnectionGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innovation: int

@dataclass
class Genome:
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    conns: Dict[int, ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    _phenotype: Optional[Callable[[List[float]], List[float]]] = field(default=None, repr=False, compare=False)

    def copy(self) -> "Genome":
        g = Genome()
        g.nodes = {nid: NodeGene(n.id, n.type, n.activation) for nid, n in self.nodes.items()}
        g.conns = {inn: ConnectionGene(c.in_id, c.out_id, c.weight, c.enabled, c.innovation)
                   for inn, c in self.conns.items()}
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        return g