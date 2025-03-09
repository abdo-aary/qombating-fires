import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import preprocessOpt
import preprocessOpt.candidates
import preprocessOpt.utilities

file_path = "./optim/output_measure.csv"
df = pd.read_csv(file_path)
# Paramètres de la grille
grid_size = 10  # Taille de la grille
num_tours = 15  # Nombre de tournées possibles
budget = 5  # Nombre maximum de tournées à sélectionner


# Filtrer les cellules où IS_FIRE == 0
fire_free_cells = df[df["IS_FIRE"] == 0]
# Sélectionner une cellule aléatoire parmi celles avec IS_FIRE == 0
#selected_cell = fire_free_cells.sample(1).iloc[0]

#We launch drones from Montréal
selected_cell = df[(df["COORDINATES_LAT"] == 1) & (df["COORDINATES_LON"] == 25)].iloc[0]
tours = preprocessOpt.candidates.candidates_generation(selected_cell,df,800//20,1000,800,15)
graph, isolated_vertex = preprocessOpt.utilities.tours_conflict_graph(tours,2)
# Construction du graphe des tournées

G = nx.Graph()
G.add_edges_from(graph)

# Résolution du problème Maximum Independent Set avec la contrainte de budget
independent_set = nx.approximation.maximum_independent_set(G)
MIS = list(independent_set)+isolated_vertex
print("taille MIS ", len(MIS),MIS)

selected_tours = sorted(list(MIS)[:budget])

"""
# Visualisation sur une grille
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(range(grid_size + 1))
ax.set_yticks(range(grid_size + 1))
ax.grid(True, linestyle="--", linewidth=0.5)
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.invert_yaxis()

# Colorier la case de départ
start_x, start_y = tours[0][0]  # Première tournée choisie comme départ
ax.add_patch(plt.Rectangle((start_x - 0.5, start_y - 0.5), 1, 1, color="blue", alpha=0.5))

# Tracer les tournées sélectionnées
for i in selected_tours:
    path = tours[i]
    x_coords, y_coords = zip(*path)
    ax.plot(x_coords, y_coords, marker="o", linestyle="-", color="blue")

# Sauvegarder l'image
plt.savefig("./optim/res.png")
plt.show()
"""