from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from collections import deque, defaultdict
import math
import random
import os

from .genome import Genome, NodeGene, ConnectionGene, NodeType
from .problem import Problem
from .history import EvolutionHistory
from .graph_utils import has_path, topological_order_or_none
from .visualization import visualize_genome

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
# Configuration Parameters
# =========================
@dataclass
class NEATConfig:
    pop_size: int = 150
    n_inputs: int = 3
    n_outputs: int = 1
    initial_connection: str = "full"
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    compatibility_threshold: float = 3.0
    elitism: int = 2
    survival_threshold: float = 0.2
    crossover_rate: float = 0.75
    mutation_weight_perturb_prob: float = 0.9
    mutation_weight_sigma: float = 0.5
    mutation_weight_reset_prob: float = 0.1
    add_connection_prob: float = 0.05
    add_node_prob: float = 0.03
    prune_connection_prob: float = 0.02
    prune_node_prob: float = 0.01
    mutate_activation_prob: float = 0.03
    node_activation_options: List[str] = field(default_factory=lambda: ['tanh', 'sigmoid', 'relu'])
    allow_recurrent: bool = False
    max_stagnation: int = 15
    random_seed: Optional[int] = 7
    target_fitness: Optional[float] = None

# =========================
# Innovation Numbering
# =========================
@dataclass
class InnovationDB:
    conn_innovations: Dict[Tuple[int, int], int] = field(default_factory=dict)
    next_innovation: int = 1
    next_node_id: int = 1
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)

    def ensure_io_nodes(self, n_inputs: int, n_outputs: int) -> None:
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
    innov_db.ensure_io_nodes(cfg.n_inputs, cfg.n_outputs)
    population: List[Genome] = []
    default_activation = cfg.node_activation_options[0] if cfg.node_activation_options else 'tanh'

    for _ in range(cfg.pop_size):
        g = Genome()
        for nid in innov_db.input_ids:
            g.nodes[nid] = NodeGene(nid, NodeType.INPUT, 'identity')
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
    child = Genome()
    for nid, n in more_fit.nodes.items():
        child.nodes[nid] = NodeGene(n.id, n.type, n.activation)
    inno_all = set(more_fit.conns.keys()).union(less_fit.conns.keys())
    for inn in sorted(inno_all):
        g1 = more_fit.conns.get(inn)
        g2 = less_fit.conns.get(inn)
        chosen: Optional[ConnectionGene] = None
        if g1 and g2:
            chosen = random.choice([g1, g2])
            enabled = chosen.enabled
            if (not g1.enabled or not g2.enabled) and random.random() < 0.75:
                enabled = False
            child.conns[chosen.innovation] = ConnectionGene(
                chosen.in_id, chosen.out_id, chosen.weight, enabled, chosen.innovation
            )
        elif g1:
            child.conns[g1.innovation] = ConnectionGene(
                g1.in_id, g1.out_id, g1.weight, g1.enabled, g1.innovation
            )
        else:
            continue
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
        if random.random() < cfg.mutation_weight_reset_prob:
            c.weight = random.uniform(-1.0, 1.0)
        elif random.random() < cfg.mutation_weight_perturb_prob:
            c.weight += random.gauss(0.0, cfg.mutation_weight_sigma)

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
# History + Main Evolution Loop
# =========================
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