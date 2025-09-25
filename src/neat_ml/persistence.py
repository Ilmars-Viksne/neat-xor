from __future__ import annotations
import json
import os
from typing import Dict, Optional
from datetime import datetime

from .genome import Genome, NodeGene, ConnectionGene

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