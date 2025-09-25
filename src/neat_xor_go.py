from neat_xor import load_genome_json, decode_to_network

# Load once at startup
champion = load_genome_json("artifacts/champion_genome.json")
forward = decode_to_network(champion)

# Input combinations for XOR with bias
inputs = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
]

# Run and print predictions
for x in inputs:
    pred = forward(x)[0]
    print(f"  input={x} -> pred={pred:.4f}")
