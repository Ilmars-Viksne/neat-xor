from __future__ import annotations
import os
import math
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .genome import Genome, NodeType
from .history import EvolutionHistory
from .graph_utils import topological_order_or_none

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
    by_depth: Dict[int, list[int]] = defaultdict(list)
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

        if node.type == NodeType.INPUT:
            face = "tab:blue"
        else:
            face = ACT_COLOR.get(node.activation, "#CCCCCC")

        ax.scatter([x], [y], s=360, c=face, marker=marker,
                   edgecolors=fg, lw=1.2, zorder=3)

        tag = act_tag.get(node.activation, "?")
        ax.text(x, y, f"{nid}\n{tag}", fontsize=8, linespacing=0.9,
                ha="center", va="center", color=fg, zorder=5)

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
    ax.add_artist(type_leg)

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