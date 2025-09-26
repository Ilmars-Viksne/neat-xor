# NEAT-ML: NeuroEvolution of Augmenting Topologies

NEAT-ML is a Python library that implements the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. It is designed to be extensible, allowing developers to define their own problems and evolve neural networks to solve them.

This project refactors a monolithic NEAT script into a structured, installable Python package that can be distributed via `pip`.

## Features

- **Extensible Problem Definition**: Define your own fitness evaluation by subclassing the `Problem` abstract base class.
- **Structured & Packaged**: The core logic is separated into a `neat_ml` library, installable via `pip`.
- **Command-Line Interface**: A simple CLI (`neat-train`) to run training for any defined problem.
- **Visualization**: Automatically generates visualizations of champion genomes and fitness history plots.
- **Persistence**: Save and load champion genomes in a readable JSON format.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ilmars-Viksne/neat-xor
    cd neat-xor
    ```

2.  **Install the package:**
    For development, install it in editable mode. This will also install dependencies like `matplotlib`.
    ```bash
    pip install -e .
    ```

## Usage

There are two primary ways to use this library: via the command-line interface for quick experiments, or by importing it into your own Python scripts for more complex integrations.

### 1. Using the `neat-train` CLI

The CLI is the quickest way to start an evolution, provided you have defined your problem in a Python module.

**Example:** Run the included XOR problem.

```bash
neat-train examples.xor_problem.XORProblem --generations 100
```

-   `examples.xor_problem.XORProblem` is the Python import path to the class that defines the problem.
-   `--generations` specifies how many generations to run the evolution.

This command will:
- Train a model to solve the XOR problem.
- Save the champion genome to `artifacts/champion.json`.
- Save visualizations to the `viz/` directory.
- Save a fitness history plot to `artifacts/fitness_history.png`.

### 2. Using the Library in Your Code

For more control, you can import `neat_ml` directly into your scripts. This is the recommended approach for integrating NEAT into a larger application.

#### Step 1: Define Your Problem

Create a file (e.g., `my_problem.py`) and define a class that inherits from `neat_ml.Problem`. You must implement `get_input_output_size` and `evaluate`.

```python
# my_problem.py
from neat_ml import Problem

class MyCoolProblem(Problem):
    def get_input_output_size(self):
        # Number of network inputs and outputs
        return (10, 2)

    def evaluate(self, forward, genome):
        # Your custom fitness logic here...
        # The 'forward' function is the network's phenotype.
        fitness = 0.0
        # ... calculate fitness ...
        return fitness
```

#### Step 2: Write a Training Script

Create a script to configure and run the NEAT evolution.

```python
# train_my_problem.py
from neat_ml import NEATConfig, run_neat, save_genome_json
from my_problem import MyCoolProblem

# 1. Instantiate your problem
problem = MyCoolProblem()

# 2. Create a configuration
config = NEATConfig(pop_size=200, random_seed=42)

# 3. Run the evolution
champion, history = run_neat(config, problem, max_generations=200)

# 4. Save the champion
if champion:
    save_genome_json(champion, "my_champion.json")
```

The full example for the XOR problem can be found in `examples/train_xor.py`.

### 3. Running a Trained Champion

Once you have a saved champion (`.json` file), you can load it and use it for inference.

```python
from neat_ml import load_genome_json, decode_to_network

# Load the genome
champion = load_genome_json("my_champion.json")

# Decode it into a callable network
net = decode_to_network(champion)

# Use it for predictions
my_input = [0.1, 0.2, ..., 0.9]
output = net(my_input)
print(f"Prediction: {output}")
```

See `examples/run_champion.py` for a working example.