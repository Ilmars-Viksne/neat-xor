"""
Network Genome Editor - CLI (interactive REPL)

Usage:
  python cli_editor.py [optional.pkl]

Core commands:
  open <path>                 Open a .pkl genome
  save [<path>]               Save to last opened path or to <path> if given
  info                        Show brief genome info
  list nodes [--fields a,b]   List nodes (default fields: id,type,activation,x,y)
  list conns                  List connections (in -> out, weight, enabled, innovation)
  show node <id>              Show all attributes for a node
  show conn <in> <out>        Show attributes for a connection
  set node <id> <attr> <val>  Update one node field (type/activation)
  set conn <in> <out> <attr> <val>  Update connection (weight/enabled)
  move node <id> <x> <y>      Move a node to new coordinates
  add node <id> [k=v ...]     Add node with optional type=, activation=
  del node <id>               Delete node (and attached connections)
  add conn <in> <out> [k=v...] Add connection with weight=, enabled=, innovation=
  del conn <in> <out>         Delete connection
  layout circle [--width W --height H]  Assign circular x,y
  export json <path>          Export to JSON (nodes+conns)
  import json <path>          Import from JSON (replaces current genome)
  help [cmd]                  Show help
  quit / exit / ^D            Leave the program
"""

from __future__ import annotations
import cmd
import json
import math
import os
import pickle
import shlex
import sys
from typing import Dict, List, Tuple

# Fix imports for running from examples/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neat_ml.genome import Genome, NodeGene, ConnectionGene, NodeType


# -----------------------
# Helpers / conversion
# -----------------------
TRUE_SET = {"true", "t", "1", "y", "yes", "on"}
FALSE_SET = {"false", "f", "0", "n", "no", "off"}

def to_bool(s: str) -> bool:
    ls = s.strip().lower()
    if ls in TRUE_SET: return True
    if ls in FALSE_SET: return False
    raise ValueError(f"Not a boolean: {s!r}")

def auto_cast(s: str, original_type=None):
    """Cast string to a good type: bool/int/float/str or preserve original_type."""
    if original_type is bool:
        return to_bool(s)
    if original_type in (int, float, str):
        return original_type(s)
    # best-effort auto
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # try bool keywords
    ls = s.strip().lower()
    if ls in TRUE_SET or ls in FALSE_SET:
        return to_bool(ls)
    return s

def fields_for_nodes(default=False, user_csv=None):
    if user_csv:
        return [f.strip() for f in user_csv.split(",") if f.strip()]
    if default:
        return ["id", "type", "activation", "x", "y"]
    return ["id", "type", "activation", "x", "y"]

def fmt_bool(b) -> str:
    return "true" if bool(b) else "false"


# -----------------------
# Layout
# -----------------------
def circular_layout(genome: Genome, width=1000, height=700):
    """Assigns x,y in a circle."""
    if not genome or not genome.nodes:
        return
    cx, cy = width / 2.0, height / 2.0
    r = max(60, min(cx, cy) * 0.75)
    N = len(genome.nodes)
    for i, node in enumerate(genome.nodes.values()):
        a = 2 * math.pi * i / max(1, N)
        # We dynamically attach x,y to NodeGene for visualization
        node.x, node.y = cx + r * math.cos(a), cy + r * math.sin(a)


# -----------------------
# JSON I/O
# -----------------------
def genome_to_json(genome: Genome) -> dict:
    return {
        "fitness": getattr(genome, "fitness", None),
        "nodes": [
            {
                "id": int(n.id),
                "type": str(n.type),
                "activation": str(n.activation),
                "x": float(getattr(n, "x", 0.0)),
                "y": float(getattr(n, "y", 0.0)),
            }
            for n in genome.nodes.values()
        ],
        "conns": [
            {
                "in": int(c.in_id),
                "out": int(c.out_id),
                "weight": float(c.weight),
                "enabled": bool(c.enabled),
                "innovation": int(c.innovation),
            }
            for c in genome.conns.values()
        ],
    }

def json_to_genome(data: dict) -> Genome:
    g = Genome()
    g.fitness = data.get("fitness")
    g.nodes = {}
    for nd in data.get("nodes", []):
        n = NodeGene(
            nd["id"],
            type=str(nd.get("type", NodeType.HIDDEN)),
            activation=str(nd.get("activation", "tanh")),
        )
        # Dynamically attach x,y for visualization
        n.x = float(nd.get("x", 0.0))
        n.y = float(nd.get("y", 0.0))
        g.nodes[n.id] = n
    g.conns = {}
    for cd in data.get("conns", []):
        c = ConnectionGene(
            int(cd["in"]),
            int(cd["out"]),
            weight=float(cd.get("weight", 1.0)),
            enabled=bool(cd.get("enabled", True)),
            innovation=int(cd["innovation"]),
        )
        g.conns[c.innovation] = c
    return g


# -----------------------
# REPL
# -----------------------
class GenomeCLI(cmd.Cmd):
    intro = "Network Genome Editor (CLI). Type 'help' or 'help <cmd>' for help. Type 'quit' to exit."
    prompt = "genome> "

    def __init__(self, genome: Genome | None = None, filepath: str | None = None):
        super().__init__()
        self.genome = genome or Genome()
        self.filepath = filepath

    # ---- Utility printing ----
    def _require_genome(self):
        if not self.genome or (not self.genome.nodes and not self.genome.conns):
            print("No genome loaded.")
            return False
        return True

    def _print_nodes(self, fields: List[str]):
        rows = []
        for nid in sorted(self.genome.nodes.keys()):
            n = self.genome.nodes[nid]
            row = []
            for f in fields:
                v = getattr(n, f, None) if f != "id" else nid
                if isinstance(v, bool):
                    v = fmt_bool(v)
                row.append(str(v))
            rows.append(row)
        # simple column widths
        widths = [max(len(fields[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(fields))]
        head = "  ".join(f.ljust(widths[i]) for i in range(len(fields)))
        print(head)
        print("-" * len(head))
        for r in rows:
            print("  ".join(r[i].ljust(widths[i]) for i in range(len(fields))))

    def _print_conns(self):
        rows = []
        for c in self.genome.conns.values():
            rows.append([str(c.in_id), "->", str(c.out_id),
                         f"{c.weight:.3f}", fmt_bool(c.enabled), str(c.innovation)])
        fields = ["in", "", "out", "weight", "enabled", "innovation"]
        widths = [max(len(fields[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(fields))]
        head = "  ".join(f.ljust(widths[i]) for i in range(len(fields)))
        print(head)
        print("-" * len(head))
        for r in rows:
            print("  ".join(r[i].ljust(widths[i]) for i in range(len(fields))))

    def _find_conn(self, a: int, b: int) -> ConnectionGene | None:
        for c in self.genome.conns.values():
            if c.in_id == a and c.out_id == b:
                return c
        return None

    # ---- Commands ----
    def do_open(self, arg):
        """open <path>
        Open a pickle (.pkl) genome."""
        args = shlex.split(arg)
        if not args:
            print("Usage: open <path.pkl>")
            return
        path = args[0]
        try:
            with open(path, "rb") as f:
                self.genome = pickle.load(f)
            self.filepath = path
            print(f"Opened {path}")
            self.do_info("")
        except Exception as e:
            print(f"ERROR: could not open: {e}")

    def do_save(self, arg):
        """save [<path>]
        Save genome to last opened path, or to <path> if provided."""
        args = shlex.split(arg)
        path = args[0] if args else self.filepath
        if not path:
            print("Usage: save <path.pkl> (no previous path)")
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(self.genome, f)
            self.filepath = path
            print(f"Saved to {path}")
        except Exception as e:
            print(f"ERROR: could not save: {e}")

    def do_info(self, _):
        """info
        Show brief info: counts, filepath, fitness."""
        if not self.genome:
            print("No genome in memory.")
            return
        n_nodes = len(self.genome.nodes)
        n_conns = len(self.genome.conns)
        print(f"File: {self.filepath or '(unsaved)'}")
        print(f"Nodes: {n_nodes}   Connections: {n_conns}")
        print(f"Fitness: {getattr(self.genome, 'fitness', None)}")

    def do_list(self, arg):
        """list nodes [--fields a,b,c] | list conns
        List nodes or connections."""
        args = shlex.split(arg)
        if not args:
            print("Usage: list nodes [--fields id,type,activation,x,y] | list conns")
            return
        if args[0] == "nodes":
            fields = fields_for_nodes(default=True)
            # parse optional --fields=something
            for tok in args[1:]:
                if tok.startswith("--fields="):
                    fields = fields_for_nodes(user_csv=tok.split("=", 1)[1])
            self._print_nodes(fields)
        elif args[0] == "conns":
            self._print_conns()
        else:
            print("Usage: list nodes [...] | list conns")

    def do_show(self, arg):
        """show node <id> | show conn <in> <out>"""
        args = shlex.split(arg)
        if len(args) < 2:
            print("Usage: show node <id> | show conn <in> <out>")
            return
        if args[0] == "node":
            try:
                nid = int(args[1])
                n = self.genome.nodes[nid]
                for k in [a for a in dir(n) if not a.startswith("__") and not callable(getattr(n, a))]:
                    print(f"{k}: {getattr(n, k)}")
            except Exception:
                print(f"Node {args[1]} not found.")
        elif args[0] == "conn":
            if len(args) < 3:
                print("Usage: show conn <in> <out>")
                return
            a, b = int(args[1]), int(args[2])
            c = self._find_conn(a, b)
            if not c:
                print(f"Connection {a}->{b} not found.")
                return
            for k in [a for a in dir(c) if not a.startswith("__") and not callable(getattr(c, a))]:
                print(f"{k}: {getattr(c, k)}")
        else:
            print("Usage: show node <id> | show conn <in> <out>")

    def do_set(self, arg):
        """set node <id> <attr> <value>
           set conn <in> <out> <attr> <value>"""
        args = shlex.split(arg)
        if not args:
            print(self.do_set.__doc__)
            return
        if args[0] == "node" and len(args) >= 4:
            nid = int(args[1]); attr = args[2]; val = " ".join(args[3:])
            if nid not in self.genome.nodes:
                print(f"Node {nid} not found.")
                return
            n = self.genome.nodes[nid]
            if not hasattr(n, attr):
                print(f"Node has no attribute '{attr}'.")
                return
            orig_t = type(getattr(n, attr))
            try:
                setattr(n, attr, auto_cast(val, orig_t))
                print(f"Node {nid}.{attr} = {getattr(n, attr)}")
            except Exception as e:
                print(f"ERROR: {e}")
        elif args[0] == "conn" and len(args) >= 5:
            a, b = int(args[1]), int(args[2]); attr = args[3]; val = " ".join(args[4:])
            c = self._find_conn(a, b)
            if not c:
                print(f"Connection {a}->{b} not found.")
                return
            if not hasattr(c, attr):
                print(f"Connection has no attribute '{attr}'.")
                return
            orig_t = type(getattr(c, attr))
            try:
                setattr(c, attr, auto_cast(val, orig_t))
                print(f"Conn {a}->{b}.{attr} = {getattr(c, attr)}")
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            print(self.do_set.__doc__)

    def do_move(self, arg):
        """move node <id> <x> <y>"""
        args = shlex.split(arg)
        if len(args) != 4 or args[0] != "node":
            print(self.do_move.__doc__)
            return
        nid = int(args[1]); x = float(args[2]); y = float(args[3])
        if nid not in self.genome.nodes:
            print(f"Node {nid} not found.")
            return
        n = self.genome.nodes[nid]
        n.x, n.y = x, y
        print(f"Node {nid} moved to ({x:.1f}, {y:.1f})")

    def do_add(self, arg):
        """add node <id> [type=.. activation=..]
           add conn <in> <out> [weight=.. enabled=true|false innovation=..]"""
        args = shlex.split(arg)
        if not args:
            print(self.do_add.__doc__)
            return
        if args[0] == "node" and len(args) >= 2:
            nid = int(args[1])
            if nid in self.genome.nodes:
                print(f"Node {nid} already exists.")
                return
            # defaults
            kwargs = {"type": NodeType.HIDDEN, "activation": "tanh"}
            for tok in args[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k in ("type", "activation"):
                        kwargs[k] = v
            n = NodeGene(nid, **kwargs)
            n.x, n.y = 0.0, 0.0 # Attach for viz
            self.genome.nodes[nid] = n
            print(f"Added node {nid} ({kwargs})")
        elif args[0] == "conn" and len(args) >= 3:
            a, b = int(args[1]), int(args[2])
            if a not in self.genome.nodes or b not in self.genome.nodes:
                print("Both endpoint nodes must exist.")
                return
            if self._find_conn(a, b):
                print(f"Connection {a}->{b} already exists.")
                return
            # Find a new innovation number
            innov = max(self.genome.conns.keys(), default=-1) + 1
            weight = 1.0; enabled = True
            for tok in args[3:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k == "weight":
                        weight = float(v)
                    elif k == "enabled":
                        enabled = to_bool(v)
                    elif k == "innovation":
                        innov = int(v)

            if innov in self.genome.conns:
                print(f"Innovation {innov} already exists.")
                return

            c = ConnectionGene(a, b, weight=weight, enabled=enabled, innovation=innov)
            self.genome.conns[innov] = c
            print(f"Added connection {a}->{b} (innov={innov}, weight={weight}, enabled={enabled})")
        else:
            print(self.do_add.__doc__)

    def do_del(self, arg):
        """del node <id> | del conn <in> <out>"""
        args = shlex.split(arg)
        if not args:
            print(self.do_del.__doc__)
            return
        if args[0] == "node" and len(args) == 2:
            nid = int(args[1])
            if nid not in self.genome.nodes:
                print(f"Node {nid} not found.")
                return
            # remove attached connections
            before = len(self.genome.conns)
            self.genome.conns = {innov: c for innov, c in self.genome.conns.items()
                                 if c.in_id != nid and c.out_id != nid}
            removed = before - len(self.genome.conns)
            del self.genome.nodes[nid]
            print(f"Deleted node {nid} and {removed} attached connection(s).")
        elif args[0] == "conn" and len(args) == 3:
            a, b = int(args[1]), int(args[2])
            c = self._find_conn(a, b)
            if not c:
                print(f"Connection {a}->{b} not found.")
                return
            del self.genome.conns[c.innovation]
            print(f"Deleted connection {a}->{b}.")
        else:
            print(self.do_del.__doc__)

    def do_layout(self, arg):
        """layout circle [--width W --height H]
        Assign circular x,y coordinates."""
        args = shlex.split(arg)
        if not args or args[0] != "circle":
            print(self.do_layout.__doc__)
            return
        width, height = 1000, 700
        for tok in args[1:]:
            if tok.startswith("--width"):
                _, _, v = tok.partition("=")
                width = float(v) if v else width
            elif tok.startswith("--height"):
                _, _, v = tok.partition("=")
                height = float(v) if v else height
        circular_layout(self.genome, width, height)
        print(f"Laid out {len(self.genome.nodes)} nodes on a circle ({width}x{height}).")

    def do_export(self, arg):
        """export json <path>
        Export genome to JSON."""
        args = shlex.split(arg)
        if len(args) != 2 or args[0] != "json":
            print(self.do_export.__doc__)
            return
        path = args[1]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(genome_to_json(self.genome), f, indent=2)
            print(f"Exported JSON to {path}")
        except Exception as e:
            print(f"ERROR: {e}")

    def do_import(self, arg):
        """import json <path>
        Import genome from JSON (replaces current genome)."""
        args = shlex.split(arg)
        if len(args) != 2 or args[0] != "json":
            print(self.do_import.__doc__)
            return
        path = args[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.genome = json_to_genome(data)
            self.filepath = None
            print(f"Imported JSON from {path}")
            self.do_info("")
        except Exception as e:
            print(f"ERROR: {e}")

    def do_help(self, arg):
        """help [cmd]"""
        if arg:
            try:
                func = getattr(self, 'do_' + arg)
                if func.__doc__:
                    print(func.__doc__)
                else:
                    super().do_help(arg)
            except AttributeError:
                super().do_help(arg)
        else:
            print(__doc__)
            print("Common commands: open, save, info, list, show, set, move, add, del, layout, export, import, quit")

    # Exit commands
    def do_quit(self, _):  # quit
        """quit
        Exit the editor."""
        print("Bye.")
        return True

    def do_exit(self, _):  # exit
        """exit
        Exit the editor."""
        return self.do_quit(_)

    def do_EOF(self, _):   # Ctrl-D
        print()
        return True


# -----------------------
# Dummy for first run
# -----------------------
def ensure_dummy(path="dummy_champion.pkl") -> str:
    if os.path.exists(path):
        return path
    print(f"Creating a dummy file for testing: {path}")
    g = Genome()
    g.fitness = 42.0
    g.nodes = {
        0: NodeGene(0, type=NodeType.INPUT),
        1: NodeGene(1, type=NodeType.INPUT),
        2: NodeGene(2, type=NodeType.OUTPUT, activation='relu'),
        3: NodeGene(3, type=NodeType.HIDDEN),
    }
    g.conns = {
        0: ConnectionGene(0, 2, weight=1.5, enabled=True, innovation=0),
        1: ConnectionGene(1, 2, weight=-0.8, enabled=False, innovation=1),
        2: ConnectionGene(2, 3, weight=0.9, enabled=True, innovation=2),
        3: ConnectionGene(0, 3, weight=0.5, enabled=True, innovation=3),
    }
    # give them initial positions via a circle
    circular_layout(g)
    with open(path, "wb") as f:
        pickle.dump(g, f)
    return path


# -----------------------
# Entrypoint
# -----------------------
def main():
    genome = None
    filepath = None
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
        try:
            with open(filepath, "rb") as f:
                genome = pickle.load(f)
        except Exception as e:
            print(f"Could not open '{filepath}': {e}")
            genome, filepath = None, None
    else:
        # auto-create & open a dummy file on first run
        p = ensure_dummy()
        with open(p, "rb") as f:
            genome = pickle.load(f)
        filepath = p

    cli = GenomeCLI(genome, filepath)
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")

if __name__ == "__main__":
    main()