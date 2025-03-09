import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import preprocessOpt
import preprocessOpt.candidates
import preprocessOpt.utilities
import modelQuantum

import sys
import os

# Add the parent directory of the current file to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

file_path = "storage/output/test_samples_measures.csv"
df = pd.read_csv(file_path)


#At most we expect the drone to do 3 tours a day
budget = 3

#We launch drones from Montréal
selected_cell = df[(df["COORDINATES_LAT"] == 1) & (df["COORDINATES_LON"] == 25)].iloc[0]

#We launch drones from Saguenay#
#selected_cell = df[(df["COORDINATES_LAT"] == 13) & (df["COORDINATES_LON"] == 33)].iloc[0]

#We launch drones from Gaspée
#selected_cell = df[(df["COORDINATES_LAT"] == 14) & (df["COORDINATES_LON"] == 60)].iloc[0]

tours = preprocessOpt.candidates.candidates_generation(selected_cell,df,800//20,1000,800,15)
graph, isolated_vertex = preprocessOpt.utilities.tours_conflict_graph(tours,2)

###Classique###

G1 = nx.Graph()
G1.add_edges_from(graph)

# Résolution du problème Maximum Independent Set avec la contrainte de budget
independent_set = nx.approximation.maximum_independent_set(G1)
MIS = list(independent_set)+isolated_vertex
print("taille MIS ", len(MIS),MIS)

selected_tours = sorted(list(MIS)[:budget])
res= [tours[i][0] for i in selected_tours]

# Convert the list of DataFrames into a single DataFrame
df = pd.concat(res, ignore_index=True)

# Save to CSV
df.to_csv("resultsCMtl.csv", index=False)

###QUANTIQUE###


qpu_min_dist = 5.0  # Example minimum distance constraint of the QPU, tiré de Analog Device
qpu_max_dist = 38.0  # Example maximum distance constraint of the QPU, tiré de Analog Device
mis,pos = modelQuantum.solve_mis_with_pulser(G1, qpu_min_dist, qpu_max_dist)
print("Maximum Independent Set:", mis)

MIS = list(mis)+isolated_vertex
print("taille MIS ", len(MIS),MIS)


selected_tours = sorted(list(MIS)[:budget])

res= [tours[i][0] for i in selected_tours]

# Convert the list of DataFrames into a single DataFrame
df = pd.concat(res, ignore_index=True)

# Save to CSV
df.to_csv("resultsQMtl.csv", index=False)