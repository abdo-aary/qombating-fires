def calculate_tour_distance(start_cell, abstract_tour_moves, visited_cells_df):
    """
    Calculates the total distance of a surveillance tour based on cell dimensions and movement directions.
    """
    total_distance = 0.0

    # Get starting cell width and length
    start_width = start_cell["CELL_WIDTH"]
    start_length = start_cell["CELL_LENGTH"]

    # Add half the starting cell width/length based on the first move
    if abstract_tour_moves[0] in ["E", "W"]:
        total_distance += start_width / 2
    else:
        total_distance += start_length / 2

    # Iterate through the moves to sum distances
    for i in range(len(abstract_tour_moves) - 1):
        current_cell = visited_cells_df.iloc[i]
        if abstract_tour_moves[i] in ["E", "W"]:
            total_distance += current_cell["CELL_WIDTH"]
        else:
            total_distance += current_cell["CELL_LENGTH"]

    # Add half the final cell's width/length based on the last move
    last_cell = visited_cells_df.iloc[-1]
    if abstract_tour_moves[-1] in ["E", "W"]:
        total_distance += last_cell["CELL_WIDTH"] / 2
    else:
        total_distance += last_cell["CELL_LENGTH"] / 2

    return total_distance

def tour_real_distance_checking(start_cell, abstract_tour_moves, visited_cells_df,max_length):
    dist = calculate_tour_distance(start_cell, abstract_tour_moves, visited_cells_df)
    if dist < max_length :
        return True
    else :
        return False


def tours_pair_conflict_detection(tour_df1, tour_df2):
    """
    Checks if two tours share at least one common cell.
    Returns True if they share at least one row, otherwise False.
    """
    # Extract sets of unique (COORDINATES_LAT, COORDINATES_LON) pairs from both tours
    cells_tour1 = set(zip(tour_df1[0]["COORDINATES_LAT"], tour_df1[0]["COORDINATES_LON"]))
    cells_tour2 = set(zip(tour_df2[0]["COORDINATES_LAT"], tour_df2[0]["COORDINATES_LON"]))

    # Check if there is any intersection (shared cell) between the two sets
    return not cells_tour1.isdisjoint(cells_tour2)


def tours_conflict_graph(pool,min_shared_cells):
    """
    Given a list of tour DataFrames, finds all pairs of tours that share at least `min_shared_cells`.
    Returns a list of tuples (i, j) where tours[i] and tours[j] share at least `min_shared_cells`.
    Also returns a list of isolated tours that do not share `min_shared_cells` with any other tour.
    """
    shared_pairs = set()
    isolate = set(range(len(pool)))  # Assume all are isolated initially

    # Compare each pair of tours
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):  # Ensure (i, j) ordering (no duplicates)
            # Extract unique (lat, lon) pairs for both tours
            cells_tour1 = set(zip(pool[i][0]["COORDINATES_LAT"], pool[i][0]["COORDINATES_LON"]))
            cells_tour2 = set(zip(pool[j][0]["COORDINATES_LAT"], pool[j][0]["COORDINATES_LON"]))

            # Find the number of shared cells
            shared_count = len(cells_tour1 & cells_tour2)

            if shared_count >= min_shared_cells:
                shared_pairs.add((i, j))
                isolate.discard(i)  # Not isolated if it shares enough cells with another tour
                isolate.discard(j)

    return list(shared_pairs), list(isolate)