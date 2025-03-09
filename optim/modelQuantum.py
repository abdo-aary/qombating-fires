import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pulser
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser_simulation import QutipEmulator
import preprocessOpt
import pandas as pd
from preprocessOpt import candidates
from preprocessOpt import utilities
from collections import Counter


def solve_mis_with_pulser(graph, qpu_min_dist, qpu_max_dist, max_attempts=100000):
    def compute_distances(positions):
        """Compute all pairwise distances between nodes in the graph."""
        distances = []
        nodes = list(positions.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pos1, pos2 = np.array(positions[node1]), np.array(positions[node2])
                distances.append(np.linalg.norm(pos1 - pos2))
        return distances

    def scale_positions(positions, scale_factor):
        """Scale node positions by a given factor."""
        return {node: tuple(np.array(pos) * scale_factor) for node, pos in positions.items()}

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        positions = nx.spring_layout(graph)  # Compute initial layout
        distances = compute_distances(positions)
        min_dist, max_dist = min(distances), max(distances)

        # Check if the layout fits within the QPU constraints
        if qpu_min_dist <= min_dist and max_dist <= qpu_max_dist:
            break

        # Try scaling the layout
        scale_factor = min(qpu_min_dist / min_dist, qpu_max_dist / max_dist)
        positions = scale_positions(positions, scale_factor)

        # Recompute distances after scaling
        distances = compute_distances(positions)
        min_dist, max_dist = min(distances), max(distances)

        if qpu_min_dist <= min_dist and max_dist <= qpu_max_dist:
            break
    else:
        raise ValueError("Failed to fit the graph layout within QPU constraints after multiple attempts.")

    # Draw the graph with the computed positions
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos=positions, with_labels=True, node_size=500, node_color='skyblue')
    plt.title("Graph Layout within QPU Constraints")
    plt.show()

    # Create a Pulser register using the computed positions
    register = Register(dict(positions))

    # Define the Pulser sequence
    sequence = Sequence(register, AnalogDevice)
    sequence.declare_channel('rydberg', 'rydberg_global')

    # Define a simple Rydberg pulse (values can be tuned)
    pulse = Pulse.ConstantPulse(duration=5000, amplitude=4.0, detuning=-1.5, phase=0)
    sequence.add(pulse, 'rydberg')
    sequence.measure()

    # Run the simulation using QutipEmulator
    emulator = QutipEmulator.from_sequence(sequence)
    results = emulator.run()
    # Sample the final state to obtain measurement outcomes
    num_samples = 5000  # Number of measurement samples
    samples = results.sample_final_state(N_samples=num_samples)
    # Debugging: Print sampled results
    #print("Sampled Bitstrings:", samples)

    # Determine the most frequent bitstring
    most_common_bitstring, _ = samples.most_common(1)[0]

    # Extract the independent set from the most common bitstring
    independent_set = [node for node, bit in zip(register.qubits, most_common_bitstring) if bit == '1']

    return independent_set, positions

# Example usage:
file_path = "./optim/preprocessOpt/outputTest.csv"
df = pd.read_csv(file_path)
# Paramètres de la grille
grid_size = 10  # Taille de la grille
num_tours = 15  # Nombre de tournées possibles
budget = 5  # Nombre maximum de tournées à sélectionner


# Filtrer les cellules où IS_FIRE == 0
fire_free_cells = df[df["IS_FIRE"] == 0]
# Sélectionner une cellule aléatoire parmi celles avec IS_FIRE == 0
selected_cell = fire_free_cells.sample(1).iloc[0]
tours = candidates.candidates_generation(selected_cell,df,800//20,1000,800,15)
graph, isolated_vertex = utilities.tours_conflict_graph(tours,(800//20)//5)
# Construction du graphe des tournées

G = nx.Graph()
G.add_edges_from(graph)
qpu_min_dist = 5.0  # Example minimum distance constraint of the QPU
qpu_max_dist = 38.0  # Example maximum distance constraint of the QPU
mis,pos = solve_mis_with_pulser(G, qpu_min_dist, qpu_max_dist)
print("Maximum Independent Set:", mis)

def plot_graph_with_mis(graph, mis_nodes, positions):
    """
    Plots the graph with nodes in the Maximum Independent Set (MIS) highlighted in red.
    """
    plt.figure(figsize=(8, 8))
    
    # Color all nodes in default blue, then update MIS nodes to red
    node_colors = ['red' if node in mis_nodes else 'skyblue' for node in graph.nodes()]
    
    # Draw the graph
    nx.draw(graph, pos=positions, with_labels=True, node_size=500, node_color=node_colors, edge_color='gray')
    
    # Title
    plt.title("Graph with Maximum Independent Set Highlighted")
    
    # Show the graph
    plt.show()


# Display the graph with MIS nodes in red
plot_graph_with_mis(G, mis, pos)