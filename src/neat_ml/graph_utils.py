from __future__ import annotations
from collections import deque, defaultdict
from typing import List, Optional

from .genome import Genome

def has_path(genome: Genome, src: int, dst: int) -> bool:
    """Checks if a path exists from src to dst in the enabled connection graph."""
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
    """
    Performs a topological sort on the genome's nodes.
    Returns the sorted list of node IDs if the graph is a DAG, otherwise None.
    """
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