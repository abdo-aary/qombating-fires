import random
import pandas as pd

def compute_cell_distance(current_cell, neighbor, direction):
    """
    Compute the distance between the current cell and a neighboring cell.
    The distance is computed as:
    - (1/2 * CELL_WIDTH of current) + (1/2 * CELL_WIDTH of neighbor) for East/West moves
    - (1/2 * CELL_LENGTH of current) + (1/2 * CELL_LENGTH of neighbor) for North/South moves
    """
    if direction in ['N', 'S']:  # Moving North or South
        return (current_cell["CELL_LENGTH"] / 2) + (neighbor["CELL_LENGTH"] / 2)
    elif direction in ['W', 'E']:  # Moving West or East
        return (current_cell["CELL_WIDTH"] / 2) + (neighbor["CELL_WIDTH"] / 2)
    return 0

def generate_balanced_tour(max_length):
    if max_length < 4:
        raise ValueError("Maximum length must be at least 4.")

    # Ensure the tour length is even so it can be balanced
    tour_length = random.randint(4, max_length)
    if tour_length % 2 != 0:
        tour_length -= 1  # Ensure an even number for balancing

    # Half the moves must be in one axis (north/south), half in the other (west/east)
    half_length = tour_length // 2
    north_south_moves = ["N"] * (half_length // 2) + ["S"] * (half_length // 2)
    west_east_moves = ["W"] * (half_length // 2) + ["E"] * (half_length // 2)

    # Combine both lists and shuffle for randomness
    moves = north_south_moves + west_east_moves
    random.shuffle(moves)
    
    return moves



def generate_cell_tour(start_cell, df, max_moves):
    """
    Generate a valid tour that starts and ends at the same cell.
    Ensures every 'N' has a 'S' and every 'E' has a 'W' to form a loop.
    Returns a valid tour path and corresponding DataFrame.
    """
    start_lat = start_cell["COORDINATES_LAT"]
    start_lon = start_cell["COORDINATES_LON"]

    # Track current position
    current_lat = start_lat
    current_lon = start_lon

    # Track moves
    move_sequence = []
    visited_cells = [start_cell]

    # Convert coordinate pairs for quick lookup
    coordinate_pairs = set(zip(df["COORDINATES_LAT"], df["COORDINATES_LON"]))

    # Movement mapping
    move_mapping = {
        "N": (1, 0),  # Move North (Increase LAT)
        "S": (-1, 0), # Move South (Decrease LAT)
        "E": (0, 1),  # Move East (Increase LON)
        "W": (0, -1)  # Move West (Decrease LON)
    }

    # Ensure balance in movements
    move_balance = {"N": 0, "S": 0, "E": 0, "W": 0}

    for _ in range(max_moves):
        # Determine valid moves from current position
        valid_moves = [
            move for move, (lat_offset, lon_offset) in move_mapping.items()
            if (current_lat + lat_offset, current_lon + lon_offset) in coordinate_pairs
        ]

        if not valid_moves:
            break  # No valid moves, exit early

        # Choose a random valid move
        chosen_move = random.choice(valid_moves)

        # Update position
        lat_offset, lon_offset = move_mapping[chosen_move]
        current_lat += lat_offset
        current_lon += lon_offset

        # Update move balance
        move_balance[chosen_move] += 1

        # Append move to sequence
        move_sequence.append(chosen_move)

        # Retrieve and store the new cell
        new_cell = df[
            (df["COORDINATES_LAT"] == current_lat) & 
            (df["COORDINATES_LON"] == current_lon)
        ].iloc[0]

        visited_cells.append(new_cell)

        # Check if the tour is complete (balanced in all directions and back to start)
        if (
            move_balance["N"] == move_balance["S"] and
            move_balance["E"] == move_balance["W"] and
            current_lat == start_lat and current_lon == start_lon
        ):
            return pd.DataFrame(visited_cells), move_sequence

    return pd.DataFrame(), move_sequence  # Return empty if tour is incomplete


def generate_tour(start_cell,df,K):
    return generate_cell_tour(start_cell,df,K)