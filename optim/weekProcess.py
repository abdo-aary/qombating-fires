import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import preprocessOpt
import preprocessOpt.candidates
import preprocessOpt.utilities
import modelQuantum

dataset_path = "optim\\output_measure.csv"
df = pd.read_csv(dataset_path)

def truncate(df):
    L = []
    for i in range(17, 24):
        L.append(df[df["DATE"] == f'2020-05-{str(i)}'])
    return L

l = truncate(df)

dfC = []
dfQ = []

# On suppose que le drone peut faire 3 sorties par jour
budget = 3

# We launch drones from Montréal
selected_cell = df[(df["COORDINATES_LAT"] == 1) & (df["COORDINATES_LON"] == 25)].iloc[0]

# We launch drones from Saguenay
# selected_cell = df[(df["COORDINATES_LAT"] == 13) & (df["COORDINATES_LON"] == 33)].iloc[0]

# We launch drones from Gaspée
# selected_cell = df[(df["COORDINATES_LAT"] == 14) & (df["COORDINATES_LON"] == 60)].iloc[0]

for data in l:
    attempt_count = 0
    max_attempts = 10
    success = False

    while not success and attempt_count < max_attempts:
        try:
            attempt_count += 1

            # Generate candidate tours
            tours = preprocessOpt.candidates.candidates_generation(
                selected_cell, data, 800 // 20, 1000, 800, 15
            )
            # Check if tours is empty
            if not tours or len(tours) == 0:
                raise ValueError("Empty tours dataset encountered.")

            # Build conflict graph
            graph, isolated_vertex = preprocessOpt.utilities.tours_conflict_graph(tours, 2)

            ### Classique ###
            G1 = nx.Graph()
            G1.add_edges_from(graph)

            # Solve Maximum Independent Set with budget constraint
            independent_set = nx.approximation.maximum_independent_set(G1)
            MIS = list(independent_set) + isolated_vertex
            print("Classique - taille MIS:", len(MIS), MIS)

            selected_tours = sorted(list(MIS)[:budget])
            # Validate selected tour indices
            if not selected_tours or any(i >= len(tours) for i in selected_tours):
                raise ValueError("Selected tours indices invalid or empty for Classique.")

            res_classique = [tours[i][0] for i in selected_tours]
            df_classique = pd.concat(res_classique, ignore_index=True)
            dfC.append(df_classique)

            ### Quantique ###
            qpu_min_dist = 5.0  # minimum distance constraint for the QPU
            qpu_max_dist = 38.0  # maximum distance constraint for the QPU
            mis, pos = modelQuantum.solve_mis_with_pulser(G1, qpu_min_dist, qpu_max_dist)
            print("Quantique - Maximum Independent Set:", mis)

            MIS_quantique = list(mis) + isolated_vertex
            print("Quantique - taille MIS:", len(MIS_quantique), MIS_quantique)

            selected_tours_q = sorted(list(MIS_quantique)[:budget])
            if not selected_tours_q or any(i >= len(tours) for i in selected_tours_q):
                raise ValueError("Selected tours indices invalid or empty for Quantique.")

            res_quantique = [tours[i][0] for i in selected_tours_q]
            df_quantique = pd.concat(res_quantique, ignore_index=True)
            dfQ.append(df_quantique)

            # If both sections succeeded, exit the while loop
            success = True

        except ValueError as e:
            print(f"Attempt {attempt_count} failed with error: {e}. Retrying...")

    if not success:
        print("Max attempts reached for this day. Skipping day data processing.")

# Save results to CSV if available
if dfC:
    resultC = pd.concat(dfC, ignore_index=True)
    resultC.to_csv("resultsMtl7C.csv", index=False)
else:
    print("No valid Classique results to save.")

if dfQ:
    resultQ = pd.concat(dfQ, ignore_index=True)
    resultQ.to_csv("resultsMtl7Q.csv", index=False)
else:
    print("No valid Quantique results to save.")
