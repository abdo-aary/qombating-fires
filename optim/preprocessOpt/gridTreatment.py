import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import toursGen
import utilities
import candidates
# Charger le fichier CSV
file_path = "./optim/preprocessOpt/outputTest.csv"
df = pd.read_csv(file_path)

# Afficher un aperçu des données
df.head()

# Extraire les coordonnées des cellules
latitudes = df["CELL_LAT"].unique()
longitudes = df["CELL_LON"].unique()

# Filtrer les cellules où IS_FIRE == 0
fire_free_cells = df[df["IS_FIRE"] == 0]

# Sélectionner une cellule aléatoire parmi celles avec IS_FIRE == 0
selected_cell = fire_free_cells.sample(1).iloc[0]


# Generate a tour with max distance K (e.g., 50 km)
K = 800//20  # Adjust this as needed
#tour_path, abstract_tour= toursGen.generate_tour(selected_cell, df, K)
#print(candidates.candidates_pool(selected_cell,df,K,100))
#pool = candidates.candidates_pool(selected_cell,df,K,700)
#poolLen = candidates.candidate_length_verification(selected_cell,pool, 500)
#poolBig = candidates.small_tour_elimination(poolLen)
#selection =candidates.candidate_selection(poolBig, 0.1)
genPool = candidates.candidates_generation(selected_cell,df,K,1000,800,15)
print(genPool)
print(utilities.tours_conflict_graph(genPool,K//4))

tour_path = genPool[0][0]
visited_lats = tour_path["CELL_LAT"]
visited_lons = tour_path["CELL_LON"]

# Extract latitude and longitude of all cells for the background grid
all_lats = df["CELL_LAT"]
all_lons = df["CELL_LON"]
fire_intensity = df["IS_FIRE"]
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define color map for fire intensity
cmap = plt.cm.Reds
norm = plt.Normalize(vmin=df["IS_FIRE"].min(), vmax=df["IS_FIRE"].max())

# Plot all cells with fire intensity as color
sc = ax.scatter(all_lons, all_lats, c=fire_intensity, cmap=cmap, norm=norm, edgecolors="black", alpha=0.7)

# Plot the surveillance tour as a line
ax.plot(visited_lons, visited_lats, marker="o", linestyle="-", color="blue", markersize=5, label="Surveillance Path")

# Highlight the start and end points
ax.scatter(visited_lons.iloc[0], visited_lats.iloc[0], color="green", marker="s", s=100, label="Start Point")
#ax.scatter(visited_lons.iloc[-1], visited_lats.iloc[-1], color="red", marker="X", s=100, label="End Point")

# Add labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Grille des cellules avec intensité du feu et circuit du drone")

# Set the map limits
ax.set_xlim(df["CELL_LON"].min(), df["CELL_LON"].max())
ax.set_ylim(df["CELL_LAT"].min(), df["CELL_LAT"].max())

# Add color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, ax=ax, label="IS_FIRE")

# Add grid and legend
plt.grid(True, linestyle="--", linewidth=0.3)
ax.legend()

# Show the plot
plt.show()
