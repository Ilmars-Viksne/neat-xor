import argparse
import importlib
from datetime import datetime, timezone
import os

from . import NEATConfig, run_neat, save_genome_json, plot_history, decode_to_network
from .problem import Problem

def _import_problem_class(path: str) -> type[Problem]:
    """Dynamically imports a Problem class from a string path like 'module.ClassName'."""
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        problem_class = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import problem class '{path}'. "
                        f"Please provide a valid Python import path. Original error: {e}") from e

    if not issubclass(problem_class, Problem):
        raise TypeError(f"The class at '{path}' is not a subclass of neat_ml.Problem.")

    return problem_class


def main():
    parser = argparse.ArgumentParser(description="Train a NEAT model for a given problem.")
    parser.add_argument(
        "problem",
        type=str,
        help="The import path to the Problem class to solve (e.g., 'examples.xor_problem.XORProblem')."
    )
    parser.add_argument(
        "--save-champion",
        type=str,
        default="artifacts/champion.json",
        help="Path to save the champion genome JSON file."
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="viz",
        help="Directory to save visualization images."
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable all visualization output."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=300,
        help="Number of generations to run the evolution."
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=150,
        help="Population size for each generation."
    )

    args = parser.parse_args()

    try:
        problem_class = _import_problem_class(args.problem)
        problem_instance = problem_class()
    except (ImportError, TypeError) as e:
        parser.error(str(e))

    cfg = NEATConfig(
        pop_size=args.pop_size,
    )

    print(f"Running NEAT for problem: {args.problem}")
    print(f"Population size: {cfg.pop_size}, Generations: {args.generations}")

    champion, history = run_neat(
        cfg,
        problem_instance,
        max_generations=args.generations,
        viz_dir=None if args.no_viz else args.viz_dir,
        viz_each_gen=(not args.no_viz),
    )

    if champion is None:
        print("\nNo champion found after evolution.")
        return

    print(f"\nChampion fitness: {champion.fitness:.4f}")

    save_genome_json(
        champion,
        args.save_champion,
        meta={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "problem": args.problem,
            "generations": args.generations,
            "config": cfg.__dict__
        },
    )

    demos = list(problem_instance.demo_samples())
    if demos:
        net = decode_to_network(champion)
        print("\nChampion predictions on demo samples:")
        for x, y in demos:
            pred = net(x)[0]
            print(f"  input={x} -> pred={pred:.4f} target={y[0]:.4f}")

    if not args.no_viz:
        history_path = os.path.join(os.path.dirname(args.save_champion), 'fitness_history.png')
        plot_history(history, save_path=history_path, show=False)

if __name__ == "__main__":
    main()