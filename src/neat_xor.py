from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Callable
from abc import ABC, abstractmethod
import math
import random
from collections import deque, defaultdict
import os

import json
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl

# =========================
# Activation Functions
# =========================
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0.0, x)

def leaky_relu(x: float) -> float:
    return x if x > 0 else 0.01 * x

def identity(x: float) -> float:
    return x

ACTIVATION_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'identity': identity,
}

# =========================
# Problem Abstraction
# =========================
class Problem(ABC):
    @abstractmethod
    def get_input_output_size(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def evaluate(self, forward: Callable[[List[float]], List[float]], genome) -> float:
        ...

    def demo_samples(self) -> Iterable[Tuple[List[float], List[float]]]:
        return []
    
    # Default "not solved"
    def goal_reached(self, forward: Callable[[List[float]], List[float]], genome) -> bool:
        return False
    

class XORProblem(Problem):
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

    def evaluate(self, forward, _genome) -> float:
        sse = 0.0
        for x, y_true in self._patterns():
            y_pred = forward(x)
            y = max(0.0, min(1.0, y_pred[0]))
            sse += (y - y_true[0]) ** 2
        return 4.0 - sse

    def demo_samples(self) -> Iterable[Tuple[List[float], List[float]]]:
        return self._patterns()

    def goal_reached(self, forward, _genome) -> bool:
        tol = 1e-2  # accept predictions within 0.01 of the targets
        for x, y_true in self._patterns():
            y_pred = forward(x)[0]
            if abs(y_pred - y_true[0]) > tol:
                return False
        return True

# =========================
# Configuration Parameters
# =========================
@dataclass
class NEATConfig:
    pop_size: int = 150
    n_inputs: int = 3
    n_outputs: int = 1
    initial_connection: str = "full"

    # Compatibility distance coefficients
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    compatibility_threshold: float = 3.0

    # Selection & reproduction
    elitism: int = 2
    survival_threshold: float = 0.2
    crossover_rate: float = 0.75

    # Mutation (weights)
    mutation_weight_perturb_prob: float = 0.9   # chance to perturb (vs do nothing)
    mutation_weight_sigma: float = 0.5
    mutation_weight_reset_prob: float = 0.1     # chance to reset; applied before perturb

    # Structural mutations
    add_connection_prob: float = 0.05
    add_node_prob: float = 0.03
    prune_connection_prob: float = 0.02
    prune_node_prob: float = 0.01

    # Activation mutation
    mutate_activation_prob: float = 0.03
    node_activation_options: List[str] = field(default_factory=lambda: ['tanh', 'sigmoid', 'relu'])

    # Topology constraints
    allow_recurrent: bool = False

    # Speciation
    max_stagnation: int = 15

    random_seed: Optional[int] = 7
    
    target_fitness: Optional[float] = None

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
    activation: str = 'tanh'  # per-node activation

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
    _phenotype: Optional[Callable] = field(default=None, repr=False, compare=False)

    def copy(self) -> "Genome":
        g = Genome()
        g.nodes = {nid: NodeGene(n.id, n.type, n.activation) for nid, n in self.nodes.items()}
        g.conns = {inn: ConnectionGene(c.in_id, c.out_id, c.weight, c.enabled, c.innovation)
                   for inn, c in self.conns.items()}
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        return g

# =========================
# Innovation Numbering
# =========================
@dataclass
class InnovationDB:
    conn_innovations: Dict[Tuple[int, int], int] = field(default_factory=dict)
    next_innovation: int = 1
    next_node_id: int = 1
    # Shared (global) I/O node ids for the population:
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)

    def ensure_io_nodes(self, n_inputs: int, n_outputs: int) -> None:
        """Allocate shared input/output node IDs exactly once."""
        if not self.input_ids:
            self.input_ids = [self.allocate_node_id() for _ in range(n_inputs)]
        if not self.output_ids:
            self.output_ids = [self.allocate_node_id() for _ in range(n_outputs)]

    def get_or_create_connection_innovation(self, in_id: int, out_id: int) -> int:
        key = (in_id, out_id)
        if key not in self.conn_innovations:
            self.conn_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.conn_innovations[key]

    def allocate_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

# =========================
# Species
# =========================
@dataclass(eq=False)
class Species:
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = -float("inf")
    staleness: int = 0

# =========================
# Initialization
# =========================
def initialize_population(cfg: NEATConfig, innov_db: InnovationDB) -> List[Genome]:
    """Create initial population with shared I/O node IDs and 'full' connections."""
    innov_db.ensure_io_nodes(cfg.n_inputs, cfg.n_outputs)
    population: List[Genome] = []
    default_activation = cfg.node_activation_options[0] if cfg.node_activation_options else 'tanh'

    for _ in range(cfg.pop_size):
        g = Genome()

        # Shared input/output nodes (same IDs for every genome!)
        for nid in innov_db.input_ids:
            g.nodes[nid] = NodeGene(nid, NodeType.INPUT, 'identity')  # identity for inputs

        for nid in innov_db.output_ids:
            g.nodes[nid] = NodeGene(nid, NodeType.OUTPUT, default_activation)

        if cfg.initial_connection == "full":
            for i_id in innov_db.input_ids:
                for o_id in innov_db.output_ids:
                    inn = innov_db.get_or_create_connection_innovation(i_id, o_id)
                    w = random.uniform(-1.0, 1.0)
                    g.conns[inn] = ConnectionGene(i_id, o_id, w, True, inn)

        population.append(g)

    return population

# =========================
# Graph Utilities
# =========================
def has_path(genome: Genome, src: int, dst: int) -> bool:
    adj = defaultdict(list)
    for c in genome.conns.values():
        if c.enabled:
            adj[c.in_id].append(c.out_id)
    q = deque([src])
    visited = {src}
    while q:
        u = q.popleft()
        if u == dst:
            return True
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return False

def topological_order_or_none(genome: Genome) -> Optional[List[int]]:
    indeg = {nid: 0 for nid in genome.nodes.keys()}
    adj = defaultdict(list)
    for c in genome.conns.values():
        if c.enabled:
            adj[c.in_id].append(c.out_id)
            indeg[c.out_id] += 1

    q = deque([nid for nid, d in indeg.items() if d == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) == len(genome.nodes):
        return order
    return None

# =========================
# Compatibility Distance (δ)
# =========================
def compatibility_distance(cfg: NEATConfig, g1: Genome, g2: Genome) -> float:
    inno1 = sorted(g1.conns.keys())
    inno2 = sorted(g2.conns.keys())
    i = j = 0
    E = D = 0
    W = 0.0
    matches = 0
    max_inno1 = inno1[-1] if inno1 else 0
    max_inno2 = inno2[-1] if inno2 else 0
    N = max(len(inno1), len(inno2))
    if N < 20:
        N = 1

    while i < len(inno1) and j < len(inno2):
        a = inno1[i]
        b = inno2[j]
        if a == b:
            w1 = g1.conns[a].weight
            w2 = g2.conns[b].weight
            W += abs(w1 - w2)
            matches += 1
            i += 1
            j += 1
        elif a < b:
            if a > max_inno2: E += 1
            else: D += 1
            i += 1
        else:
            if b > max_inno1: E += 1
            else: D += 1
            j += 1

    while i < len(inno1):
        if inno1[i] > max_inno2: E += 1
        else: D += 1
        i += 1
    while j < len(inno2):
        if inno2[j] > max_inno1: E += 1
        else: D += 1
        j += 1

    W_bar = (W / matches) if matches > 0 else 0.0
    delta = (cfg.c1 * E / N) + (cfg.c2 * D / N) + (cfg.c3 * W_bar)
    return delta

# =========================
# Speciation and Fitness Sharing
# =========================
def speciate(cfg: NEATConfig, population: List[Genome], species_list: List[Species]) -> List[Species]:
    if not species_list and population:
        species_list = [Species(representative=population[0].copy())]

    for s in species_list:
        s.members.clear()

    for g in population:
        placed = False
        for s in species_list:
            if compatibility_distance(cfg, g, s.representative) < cfg.compatibility_threshold:
                s.members.append(g)
                placed = True
                break
        if not placed:
            species_list.append(Species(representative=g.copy(), members=[g]))

    for s in species_list:
        if s.members:
            s.representative = random.choice(s.members).copy()

    species_list = [s for s in species_list if s.members]
    return species_list

def adjust_fitness_and_prune(cfg: NEATConfig, species_list: List[Species]) -> None:
    for s in species_list:
        if not s.members:
            continue
        n = len(s.members)
        for g in s.members:
            g.adjusted_fitness = g.fitness / n

        best = max(m.fitness for m in s.members)
        if best > s.best_fitness:
            s.best_fitness = best
            s.staleness = 0
        else:
            s.staleness += 1

    if len(species_list) > 1:
        best_overall = max(sp.best_fitness for sp in species_list)
        species_list[:] = [
            s for s in species_list
            if s.staleness < cfg.max_stagnation or s.best_fitness == best_overall
        ]

def allocate_offspring(cfg: NEATConfig, species_list: List[Species]) -> Dict[Species, int]:
    total_adj = sum(sum(g.adjusted_fitness for g in s.members) for s in species_list)
    if total_adj == 0:
        if not species_list:
            return {}
        base = cfg.pop_size // len(species_list)
        allocation = {s: base for s in species_list}
        remainder = cfg.pop_size - base * len(species_list)
        for s in random.sample(species_list, remainder):
            allocation[s] += 1
        return allocation

    expected: List[Tuple[Species, float]] = []
    for s in species_list:
        s_adj = sum(g.adjusted_fitness for g in s.members)
        exp = (s_adj / total_adj) * cfg.pop_size
        expected.append((s, exp))

    allocation = {s: int(math.floor(exp)) for s, exp in expected}
    remainder = cfg.pop_size - sum(allocation.values())
    expected.sort(key=lambda x: x[1] - math.floor(x[1]), reverse=True)
    for i in range(remainder):
        allocation[expected[i][0]] += 1
    return allocation

# =========================
# Selection, Crossover, Mutations
# =========================
def select_parents(cfg: NEATConfig, species: Species) -> List[Genome]:
    sorted_members = sorted(species.members, key=lambda g: g.fitness, reverse=True)
    cutoff = max(1, int(len(sorted_members) * cfg.survival_threshold))
    return sorted_members[:cutoff]

def crossover(more_fit: Genome, less_fit: Genome) -> Genome:
    """
    Canonical NEAT crossover:
      - Matching genes: randomly chosen from either parent.
      - Disjoint/excess genes: inherited from the more-fit parent only.
    """
    child = Genome()

    # Start with nodes from more-fit parent
    for nid, n in more_fit.nodes.items():
        child.nodes[nid] = NodeGene(n.id, n.type, n.activation)

    # Process connections
    inno_all = set(more_fit.conns.keys()).union(less_fit.conns.keys())
    for inn in sorted(inno_all):
        g1 = more_fit.conns.get(inn)
        g2 = less_fit.conns.get(inn)

        chosen: Optional[ConnectionGene] = None
        if g1 and g2:
            chosen = random.choice([g1, g2])
            enabled = chosen.enabled
            # If either disabled, 75% chance to keep disabled (NEAT rule)
            if (not g1.enabled or not g2.enabled) and random.random() < 0.75:
                enabled = False
            child.conns[chosen.innovation] = ConnectionGene(
                chosen.in_id, chosen.out_id, chosen.weight, enabled, chosen.innovation
            )
        elif g1:
            # disjoint/excess from more-fit only
            child.conns[g1.innovation] = ConnectionGene(
                g1.in_id, g1.out_id, g1.weight, g1.enabled, g1.innovation
            )
        else:
            # gene only in less-fit is ignored (canonical NEAT)
            continue

        # Ensure node genes referenced by chosen connection exist
        last = child.conns.get(inn)
        if last:
            for nid, from_parent in [(last.in_id, g1 if g1 else g2),
                                     (last.out_id, g1 if g1 else g2)]:
                if nid not in child.nodes and from_parent:
                    node_ref = more_fit.nodes.get(nid) or less_fit.nodes.get(nid)
                    if node_ref:
                        child.nodes[nid] = NodeGene(node_ref.id, node_ref.type, node_ref.activation)

    return child

def mutate_weights(cfg: NEATConfig, g: Genome) -> None:
    for c in g.conns.values():
        if not c.enabled:
            continue
        # Reset takes precedence
        if random.random() < cfg.mutation_weight_reset_prob:
            c.weight = random.uniform(-1.0, 1.0)
        elif random.random() < cfg.mutation_weight_perturb_prob:
            c.weight += random.gauss(0.0, cfg.mutation_weight_sigma)
        else:
            # leave unchanged
            pass

def can_add_connection(cfg: NEATConfig, g: Genome, i_id: int, o_id: int) -> bool:
    if i_id == o_id:
        return False
    for c in g.conns.values():
        if c.in_id == i_id and c.out_id == o_id and c.enabled:
            return False
    if not cfg.allow_recurrent and has_path(g, o_id, i_id):
        return False
    if g.nodes[o_id].type == NodeType.INPUT:
        return False
    return True

def mutate_add_connection(cfg: NEATConfig, innov_db: InnovationDB, g: Genome) -> None:
    nodes = list(g.nodes.values())
    g._phenotype = None
    if len(nodes) < 2:
        return
    for _ in range(30):
        i = random.choice(nodes)
        o = random.choice(nodes)
        # Avoid output->input preference when feedforward
        if i.type == NodeType.OUTPUT and o.type == NodeType.INPUT:
            continue
        if can_add_connection(cfg, g, i.id, o.id):
            inn = innov_db.get_or_create_connection_innovation(i.id, o.id)
            w = random.uniform(-1.0, 1.0)
            g.conns[inn] = ConnectionGene(i.id, o.id, w, True, inn)
            return

def mutate_add_node(cfg: NEATConfig, innov_db: InnovationDB, g: Genome) -> None:
    g._phenotype = None
    enabled_conns = [c for c in g.conns.values() if c.enabled]
    if not enabled_conns:
        return
    c = random.choice(enabled_conns)
    c.enabled = False

    new_node_id = innov_db.allocate_node_id()
    default_activation = cfg.node_activation_options[0] if cfg.node_activation_options else 'tanh'
    g.nodes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN, default_activation)

    inn1 = innov_db.get_or_create_connection_innovation(c.in_id, new_node_id)
    inn2 = innov_db.get_or_create_connection_innovation(new_node_id, c.out_id)
    g.conns[inn1] = ConnectionGene(c.in_id, new_node_id, 1.0, True, inn1)
    g.conns[inn2] = ConnectionGene(new_node_id, c.out_id, c.weight, True, inn2)

def mutate_prune_connection(g: Genome) -> None:
    enabled_conns = [c for c in g.conns.values() if c.enabled]
    if not enabled_conns:
        return
    conn_to_disable = random.choice(enabled_conns)
    conn_to_disable.enabled = False
    g._phenotype = None

def mutate_prune_node(g: Genome) -> None:
    hidden_nodes = [nid for nid, n in g.nodes.items() if n.type == NodeType.HIDDEN]
    if not hidden_nodes:
        return
    node_to_prune = random.choice(hidden_nodes)
    conns_to_remove = [inn for inn, c in g.conns.items() if c.in_id == node_to_prune or c.out_id == node_to_prune]
    for inn in conns_to_remove:
        del g.conns[inn]
    del g.nodes[node_to_prune]
    g._phenotype = None

def mutate_activation(cfg: NEATConfig, g: Genome) -> None:
    """Randomly changes the activation function of a hidden or output node."""
    mutable_nodes = [n for n in g.nodes.values() if n.type != NodeType.INPUT]
    if not mutable_nodes or not cfg.node_activation_options:
        return
    node_to_mutate = random.choice(mutable_nodes)
    current_activation = node_to_mutate.activation
    possible_new = [act for act in cfg.node_activation_options if act != current_activation]
    if possible_new:
        node_to_mutate.activation = random.choice(possible_new)
        g._phenotype = None

def mutate(cfg: NEATConfig, innov_db: InnovationDB, g: Genome) -> None:
    mutate_weights(cfg, g)
    if random.random() < cfg.add_connection_prob:
        mutate_add_connection(cfg, innov_db, g)
    if random.random() < cfg.add_node_prob:
        mutate_add_node(cfg, innov_db, g)
    if random.random() < cfg.prune_connection_prob:
        mutate_prune_connection(g)
    if random.random() < cfg.prune_node_prob:
        mutate_prune_node(g)
    if random.random() < cfg.mutate_activation_prob:
        mutate_activation(cfg, g)

def reproduce(cfg: NEATConfig, innov_db: InnovationDB, species: Species, n_offspring: int) -> List[Genome]:
    if not species.members or n_offspring <= 0:
        return []
    offspring: List[Genome] = []

    # Elites (copied unchanged)
    elites = sorted(species.members, key=lambda g: g.fitness, reverse=True)[:cfg.elitism]
    for e in elites[:min(len(elites), n_offspring)]:
        offspring.append(e.copy())

    n_remaining = max(0, n_offspring - len(offspring))
    if n_remaining == 0:
        return offspring

    parents = select_parents(cfg, species)
    if not parents:
        parents = [max(species.members, key=lambda g: g.fitness)]

    for _ in range(n_remaining):
        if random.random() < cfg.crossover_rate and len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
            more_fit, less_fit = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
            child = crossover(more_fit, less_fit)
        else:
            child = random.choice(parents).copy()
        mutate(cfg, innov_db, child)
        offspring.append(child)

    return offspring

# =========================
# Network Decode & Evaluation
# =========================
def decode_to_network(genome: Genome) -> Callable[[List[float]], List[float]]:
    """
    Builds a callable forward function using the genome's structure.
    Applies the node-specific activation functions.
    """
    order = topological_order_or_none(genome)
    if order is None:
        def forward_fail(_x):
            raise RuntimeError("Genome has cycles; cannot evaluate.")
        return forward_fail

    inputs = sorted([nid for nid, n in genome.nodes.items() if n.type == NodeType.INPUT])
    outputs = sorted([nid for nid, n in genome.nodes.items() if n.type == NodeType.OUTPUT])

    incoming = defaultdict(list)
    for c in genome.conns.values():
        if c.enabled:
            incoming[c.out_id].append((c.in_id, c.weight))

    def forward(x_vec: List[float]) -> List[float]:
        if len(x_vec) != len(inputs):
            raise ValueError(f"Expected {len(inputs)} inputs, got {len(x_vec)}")
        values: Dict[int, float] = {}
        for idx, nid in enumerate(inputs):
            values[nid] = float(x_vec[idx])

        for nid in order:
            if nid in values:
                continue
            s = 0.0
            for src, w in incoming.get(nid, []):
                s += values.get(src, 0.0) * w
            activation_name = genome.nodes[nid].activation
            activation_func = ACTIVATION_FUNCTIONS.get(activation_name, identity)
            values[nid] = activation_func(s)

        return [values.get(nid, 0.0) for nid in outputs]

    return forward
def evaluate_population(population: List[Genome], problem: Problem) -> None:
    for g in population:
        if g._phenotype is None:
            g._phenotype = decode_to_network(g)
        net = g._phenotype
        try:
            g.fitness = problem.evaluate(net, g)
        except Exception:
            g.fitness = 0.0

# =========================
# Visualization
# =========================
def _compute_depths(genome: Genome) -> Dict[int, int]:
    order = topological_order_or_none(genome)
    if order is None:
        return {nid: (0 if n.type == NodeType.INPUT else 1) for nid, n in genome.nodes.items()}
    incoming = defaultdict(list)
    for c in genome.conns.values():
        if c.enabled:
            incoming[c.out_id].append(c.in_id)
    depths: Dict[int, int] = {}
    for nid in order:
        node = genome.nodes[nid]
        if node.type == NodeType.INPUT or nid not in incoming or not incoming[nid]:
            depths[nid] = 0
        else:
            depths[nid] = 1 + max(depths.get(p, 0) for p in incoming[nid])
    return depths

def visualize_genome(
    genome: Genome,
    filename: str,
    title: Optional[str] = None,
    figsize=(9, 6),
    show_weights: bool = True,
    show_disabled: bool = True,
    theme: str = "light",
    dpi: int = 180,
    weight_fmt: str = "{:+.2f}",
) -> None:
    """
    Visualize a NEAT genome.
    - Node marker shape encodes node TYPE (input/hidden/output).
    - Node fill COLOR encodes ACTIVATION function (for hidden+output).
    - Input nodes keep blue fill by default; change below if you prefer activation-based color.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    is_dark = (theme or "light").lower().startswith("d")
    bg = "#111111" if is_dark else "white"
    fg = "white" if is_dark else "black"
    label_bg = (1, 1, 1, 0.85) if not is_dark else (0.05, 0.05, 0.05, 0.85)
    label_edge = (0, 0, 0, 0.25) if not is_dark else (1, 1, 1, 0.25)

    # --- Activation color palette (light/dark tuned) ---
    act_colors_light = {
        "tanh":       "#4C78A8",  # steel blue
        "sigmoid":    "#59A14F",  # green
        "relu":       "#E45756",  # red/salmon
        "leaky_relu": "#F28E2B",  # orange
        "identity":   "#8E8E8E",  # gray
    }
    act_colors_dark = {
        "tanh":       "#6FA8DC",
        "sigmoid":    "#93C47D",
        "relu":       "#EA9999",
        "leaky_relu": "#F6B26B",
        "identity":   "#A6A6A6",
    }
    ACT_COLOR = act_colors_dark if is_dark else act_colors_light

    # Short tags for inline activation label
    act_tag = {
        "tanh": "t",
        "sigmoid": "sg",
        "relu": "rl",
        "leaky_relu": "lr",
        "identity": "id",
    }

    # ----- layout / depth computation (unchanged) -----
    depths = _compute_depths(genome)
    max_depth = max(depths.values()) if depths else 1
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for nid, d in depths.items():
        by_depth[d].append(nid)
    for d in by_depth:
        by_depth[d].sort()

    pos: Dict[int, Tuple[float, float]] = {}
    for d in range(max_depth + 1):
        nodes_at_d = by_depth.get(d, [])
        n = max(1, len(nodes_at_d))
        for i, nid in enumerate(nodes_at_d or []):
            x = 0.06 + 0.88 * (d / max(1, max_depth))
            y = 0.5 if n == 1 else 0.1 + 0.8 * (i / (n - 1))
            pos[nid] = (x, y)
    for nid, n in genome.nodes.items():
        if nid not in pos:
            x = 0.06 if n.type == NodeType.INPUT else (0.94 if n.type == NodeType.OUTPUT else 0.5)
            y = random.random() * 0.8 + 0.1
            pos[nid] = (x, y)

    # ----- edge weight colormap (unchanged) -----
    enabled_weights = [c.weight for c in genome.conns.values() if c.enabled]
    if enabled_weights:
        wmin, wmax = min(enabled_weights), max(enabled_weights)
        max_abs = max(abs(wmin), abs(wmax))
        wmin, wmax = (-1.0, 1.0) if max_abs == 0 else (-max_abs, max_abs)
    else:
        wmin, wmax = -1.0, 1.0
    norm = TwoSlopeNorm(vmin=wmin, vcenter=0.0, vmax=wmax)
    cmap = plt.get_cmap("coolwarm")

    # ----- figure -----
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    # ----- disabled edges (unchanged) -----
    if show_disabled:
        for c in genome.conns.values():
            if c.enabled:
                continue
            x1, y1 = pos[c.in_id]; x2, y2 = pos[c.out_id]
            ax.plot([x1, x2], [y1, y2],
                    color=(0.5, 0.5, 0.5, 0.35), lw=1.0, ls="--", zorder=1)

    # ----- enabled edges (unchanged) -----
    for c in genome.conns.values():
        if not c.enabled:
            continue
        x1, y1 = pos[c.in_id]; x2, y2 = pos[c.out_id]
        color = cmap(norm(c.weight))
        lw = 1.0 + 2.5 * min(1.0, abs(c.weight))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.95, zorder=2)

        if show_weights:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx, dy = (x2 - x1), (y2 - y1)
            L = math.hypot(dx, dy) + 1e-12
            nx, ny = (-dy / L, dx / L)
            off = 0.02
            lx, ly = mx + off * nx, my + off * ny
            ax.text(lx, ly, weight_fmt.format(c.weight), fontsize=8, ha="center", va="center",
                    color=fg, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc=label_bg, ec=label_edge))

    # ----- draw nodes: shape by TYPE, color by ACTIVATION -----
    # Markers by type
    marker_by_type = {
        NodeType.INPUT: "s",   # square
        NodeType.HIDDEN: "o",  # circle
        NodeType.OUTPUT: "D",  # diamond
    }

    # (1) Scatter nodes individually to allow per-node colors
    for nid, node in genome.nodes.items():
        x, y = pos[nid]
        marker = marker_by_type.get(node.type, "o")

        # Color logic:
        #  - Inputs: keep classic blue fill (clear visual anchor)
        #  - Hidden/Output: color by activation
        if node.type == NodeType.INPUT:
            face = "tab:blue"  # change to ACT_COLOR.get(node.activation, ...) if you want inputs colored by activation too
        else:
            face = ACT_COLOR.get(node.activation, "#CCCCCC")

        ax.scatter([x], [y], s=360, c=face, marker=marker,
                   edgecolors=fg, lw=1.2, zorder=3)

        # Node label: id + small activation tag (second line)
        tag = act_tag.get(node.activation, "?")
        ax.text(x, y, f"{nid}\n{tag}", fontsize=8, linespacing=0.9,
                ha="center", va="center", color=fg, zorder=5)

    # ----- legends: Node types (shapes) & Activations (colors) -----
    # Node types legend (shape only, neutral facecolor)
    type_handles = [
        Line2D([0], [0], marker=marker_by_type[NodeType.INPUT], color='none',
               markerfacecolor="none", markeredgecolor=fg, markersize=10, lw=0, label="Input"),
        Line2D([0], [0], marker=marker_by_type[NodeType.HIDDEN], color='none',
               markerfacecolor="none", markeredgecolor=fg, markersize=10, lw=0, label="Hidden"),
        Line2D([0], [0], marker=marker_by_type[NodeType.OUTPUT], color='none',
               markerfacecolor="none", markeredgecolor=fg, markersize=10, lw=0, label="Output"),
    ]
    type_leg = ax.legend(handles=type_handles, title="Node types",
                         loc="upper right", frameon=True)
    if type_leg:
        type_leg.get_frame().set_alpha(0.85)
        type_leg.get_frame().set_facecolor(label_bg)
        for txt in type_leg.get_texts():
            txt.set_color(fg)
        type_leg.get_title().set_color(fg)
    ax.add_artist(type_leg)  # keep this legend when adding the next

    # Activation legend (color patches)
    # Only include activations actually present (excluding inputs if you prefer)
    present_acts = sorted({n.activation for n in genome.nodes.values()
                           if n.type != NodeType.INPUT})
    act_handles = [
        Patch(facecolor=ACT_COLOR.get(a, "#CCCCCC"), edgecolor=fg, label=a)
        for a in present_acts
    ]
    if act_handles:
        act_leg = ax.legend(handles=act_handles, title="Activations",
                            loc="lower right", frameon=True)
        if act_leg:
            act_leg.get_frame().set_alpha(0.85)
            act_leg.get_frame().set_facecolor(label_bg)
            for txt in act_leg.get_texts():
                txt.set_color(fg)
            act_leg.get_title().set_color(fg)

    # ----- colorbar for edge weights (unchanged) -----
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Edge weight", color=fg)
    cbar.ax.yaxis.set_tick_params(color=fg)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=fg)

    if title:
        ax.set_title(title, color=fg)

    fig.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


# =========================
# History + Main Evolution Loop
# =========================
@dataclass
class EvolutionHistory:
    generations: List[int] = field(default_factory=list)
    best_overall: List[float] = field(default_factory=list)
    gen_best: List[float] = field(default_factory=list)
    avg: List[float] = field(default_factory=list)
    species: List[int] = field(default_factory=list)

def run_neat(
    cfg: NEATConfig,
    problem: Problem,
    max_generations: int = 300,
    viz_dir: Optional[str] = None,
    viz_each_gen: bool = True
) -> Tuple[Genome, EvolutionHistory]:
    cfg.n_inputs, cfg.n_outputs = problem.get_input_output_size()
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)

    innov_db = InnovationDB()
    population = initialize_population(cfg, innov_db)
    species_list: List[Species] = []
    best_overall: Optional[Genome] = None
    history = EvolutionHistory()

    if viz_dir: os.makedirs(viz_dir, exist_ok=True)

    for gen in range(1, max_generations + 1):
        evaluate_population(population, problem)
        gen_best = max(population, key=lambda g: g.fitness)
        if best_overall is None or gen_best.fitness > best_overall.fitness:
            best_overall = gen_best.copy()

        species_list = speciate(cfg, population, species_list)
        adjust_fitness_and_prune(cfg, species_list)
        allocation = allocate_offspring(cfg, species_list)

        next_population: List[Genome] = []
        for s in species_list:
            n_off = allocation.get(s, 0)
            if n_off > 0:
                next_population.extend(reproduce(cfg, innov_db, s, n_off))

        # Fill (mutated) if needed to keep population size
        if len(next_population) < cfg.pop_size and species_list:
            best_species = max(species_list, key=lambda sp: sp.best_fitness)
            while len(next_population) < cfg.pop_size and best_species.members:
                clone = random.choice(best_species.members).copy()
                mutate(cfg, innov_db, clone)
                next_population.append(clone)
        elif len(next_population) > cfg.pop_size:
            next_population = next_population[:cfg.pop_size]

        population = next_population

        avg_fit = sum(g.fitness for g in population) / max(1, len(population))
        print(f"Gen {gen:03d} Species: {len(species_list):02d} Best: {best_overall.fitness:.4f} "
              f"GenBest: {gen_best.fitness:.4f} Avg: {avg_fit:.4f}")

        history.generations.append(gen)
        history.best_overall.append(best_overall.fitness if best_overall else 0.0)
        history.gen_best.append(gen_best.fitness)
        history.avg.append(avg_fit)
        history.species.append(len(species_list))

        if viz_dir and viz_each_gen:
            fname = os.path.join(viz_dir, f"gen_{gen:03d}_champion.png")
            try:
                visualize_genome(gen_best, fname, title=f"Gen {gen} champion (fit={gen_best.fitness:.3f})")
            except Exception as e:
                print(f"[viz] Failed to save {fname}: {e}")

        # Early stopping: config threshold OR problem-specific goal
        stop = False
        if cfg.target_fitness is not None and best_overall.fitness >= cfg.target_fitness:
            print("Early stopping: target fitness reached.")
            stop = True
        else:
            try:
                net_best = decode_to_network(best_overall)
                if problem.goal_reached(net_best, best_overall):
                    print("Early stopping: problem-specific goal reached.")
                    stop = True
            except Exception:
                pass
        if stop:
            break
            
    if viz_dir and best_overall is not None:
        final_path = os.path.join(viz_dir, "final_champion.png")
        try:
            visualize_genome(best_overall, final_path, title=f"Final champion (fit={best_overall.fitness:.3f})")
        except Exception as e:
            print(f"[viz] Failed to save final visualization: {e}")

    return best_overall, history

# =========================
# Plotting
# =========================
def plot_history(history: EvolutionHistory, save_path: str = 'neat_fitness.png', show: bool = False) -> None:
    if not history.generations:
        print("No history to plot.")
        return
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(history.generations, history.best_overall, label='Best Overall', color='tab:blue', linewidth=2)
    ax1.plot(history.generations, history.gen_best, label='Gen Best', color='tab:green', alpha=0.8)
    ax1.plot(history.generations, history.avg, label='Average', color='tab:orange', alpha=0.8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (higher is better)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history.generations, history.species, label='Species count', color='tab:red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Species')

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    ax1.legend(lines, labels, loc='lower right')

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved fitness curves to {save_path}")


# =========================
# JSON Persistence (NEW)
# =========================
import json, argparse
from datetime import datetime

def genome_to_dict(genome: Genome) -> Dict:
    """Readable serialization of a genome."""
    return {
        "format_version": 1,
        "fitness": float(genome.fitness),
        "nodes": [
            {"id": n.id, "type": n.type, "activation": n.activation}
            for n in sorted(genome.nodes.values(), key=lambda x: x.id)
        ],
        "connections": [
            {
                "innovation": c.innovation,
                "in": c.in_id,
                "out": c.out_id,
                "weight": float(c.weight),
                "enabled": bool(c.enabled),
            }
            for c in sorted(genome.conns.values(), key=lambda x: x.innovation)
        ],
    }

def genome_from_dict(d: Dict) -> Genome:
    """Reconstruct a Genome from a serialized dict."""
    g = Genome()
    for n in d.get("nodes", []):
        g.nodes[int(n["id"])] = NodeGene(int(n["id"]), str(n["type"]), str(n.get("activation", "tanh")))
    for c in d.get("connections", []):
        inn = int(c["innovation"])
        g.conns[inn] = ConnectionGene(
            int(c["in"]), int(c["out"]), float(c["weight"]), bool(c["enabled"]), inn
        )
    g.fitness = float(d.get("fitness", 0.0))
    return g

def save_genome_json(genome: Genome, path: str, meta: Optional[Dict]=None) -> None:
    """Save genome + optional metadata as readable JSON."""
    payload = {"genome": genome_to_dict(genome), "meta": meta or {}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved champion genome to {path}")

def load_genome_json(path: str) -> Genome:
    """Load genome (ignoring unknown extra metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("genome", payload)
    return genome_from_dict(data)
    
    
# =========================
# CLI / Entry Point
# =========================
def main():
    problem = XORProblem(use_bias_as_input=True)
    cfg = NEATConfig(
        pop_size=150,
        initial_connection="full",
        compatibility_threshold=3.0,
        elitism=2,
        survival_threshold=0.2,
        crossover_rate=0.75,
        mutation_weight_perturb_prob=0.9,
        mutation_weight_sigma=0.5,
        mutation_weight_reset_prob=0.1,
        add_connection_prob=0.05,
        add_node_prob=0.03,
        prune_connection_prob=0.02,
        prune_node_prob=0.01,
        mutate_activation_prob=0.03,
        # You can enable more activations if you like:
        # node_activation_options=['tanh', 'sigmoid', 'relu', 'leaky_relu', 'identity'],
        node_activation_options=['relu', 'tanh'],
        allow_recurrent=False,
        max_stagnation=15,
        random_seed=7,
        target_fitness=4.0,
    )
    print("Running NEAT...")
    champion, history = run_neat(
        cfg,
        problem,
        max_generations=300,
        viz_dir='viz',
        viz_each_gen=True
    )
    print(f"\nChampion fitness: {champion.fitness:.4f}")
    demos = list(problem.demo_samples())
    if demos:
        net = decode_to_network(champion)
        print("\nChampion predictions on demo samples:")
        for x, y in demos:
            pred = net(x)[0]
            print(f" input={x} -> pred={pred:.4f} target={y[0]:.4f}")
    plot_history(history, save_path='neat_fitness.png', show=False)

    
def main():
    parser = argparse.ArgumentParser(description="NEAT XOR with JSON persistence")
    parser.add_argument("--load", type=str, default=None,
                        help="Load a saved genome JSON and run XOR demo (skip training).")
    parser.add_argument("--save", type=str, default="artifacts/champion_genome.json",
                        help="Where to save the champion genome JSON after training.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable per-generation visualization.")
    args = parser.parse_args()

    # Same XOR problem used originally
    problem = XORProblem(use_bias_as_input=True)

    if args.load:
        # Inference-only path: load champion and run immediately
        champion = load_genome_json(args.load)
        net = decode_to_network(champion)
        print(f"Loaded champion from {args.load} (fitness={champion.fitness:.4f})")
        print("\nPredictions on XOR demo samples:")
        for x, y in problem.demo_samples():
            pred = net(x)[0]
            print(f"  input={x} -> pred={pred:.4f} target={y[0]:.4f}")
        return

    # --- Training path ---
    cfg = NEATConfig(
        pop_size=150,
        initial_connection="full",
        compatibility_threshold=3.0,
        elitism=2,
        survival_threshold=0.2,
        crossover_rate=0.75,
        mutation_weight_perturb_prob=0.9,
        mutation_weight_sigma=0.5,
        mutation_weight_reset_prob=0.1,
        add_connection_prob=0.05,
        add_node_prob=0.03,
        prune_connection_prob=0.02,
        prune_node_prob=0.01,
        mutate_activation_prob=0.03,
        # You can enable more activations if you like:
        # node_activation_options=['tanh', 'sigmoid', 'relu', 'leaky_relu', 'identity'],
        node_activation_options=['relu', 'tanh'],
        allow_recurrent=False,
        max_stagnation=15,
        random_seed=7,
        target_fitness=4.0,
    )
    print("Running NEAT...")
    champion, history = run_neat(
        cfg,
        problem,
        max_generations=300,
        viz_dir='viz',
        viz_each_gen=(not args.no_viz),
    )
    print(f"\nChampion fitness: {champion.fitness:.4f}")

    # Persist the champion as readable JSON
    save_genome_json(
        champion,
        args.save,
        meta={
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "problem": "XOR",
            "use_bias_as_input": problem.use_bias_as_input,
            "n_inputs": cfg.n_inputs,
            "n_outputs": cfg.n_outputs,
            "note": "NEAT champion saved for reuse"
        },
    )

    # Quick demo
    demos = list(problem.demo_samples())
    if demos:
        net = decode_to_network(champion)
        print("\nChampion predictions on demo samples:")
        for x, y in demos:
            pred = net(x)[0]
            print(f"  input={x} -> pred={pred:.4f} target={y[0]:.4f}")

    plot_history(history, save_path='neat_fitness.png', show=False)


if __name__ == "__main__":
    main()
