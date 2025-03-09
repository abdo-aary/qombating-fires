import preprocessOpt.utilities
import preprocessOpt.toursGen

#Generate a pool of N tours candidates valid and invalid
def candidates_pool(start_cell, df, max_moves, pool_size):
    """
    Generate a list of (tour_df, move_sequence) tuples.
    If a generated tour is empty, it is ignored.
    """
    result = []
    
    while len(result) < pool_size:
        tour_df, tour_moves = preprocessOpt.toursGen.generate_cell_tour(start_cell, df, max_moves)
        
        if not tour_df.empty:  # Only add valid tours
            result.append((tour_df, tour_moves))
    
    return result

#get rid of all tours of length <4 in a pool
def small_tour_elimination(pool):
    return [(df, lst) for df, lst in pool if df.shape[0] > 4]

#Check that the tours are not longer than the maximum distance a drone can go, drop the ones that are longer
def candidate_length_verification(start_cell,pool, dMax):
    booleans = [preprocessOpt.utilities.tour_real_distance_checking(start_cell, t[1], t[0], dMax) for t in pool if not t[0].empty and len(t[1]) > 0 ]
    filtered_values =  [t for t,b in zip(pool,booleans) if b ]

    return filtered_values



#select the N-th best tours, ensure that they are not too much different 
def candidate_selection(pool,num_top_tours):
    """
    Adds the total fire probability encountered during each tour.
    Returns a list of (tour_df, move_sequence, probability_sum).
    """
    result_with_probabilities = []

    for tour_df, move_sequence in pool:
        probability_sum = tour_df["IS_FIRE"].sum()  # Sum of fire probabilities
        result_with_probabilities.append((tour_df, move_sequence, probability_sum))
        # Sort by probability_sum in ascending order

    sorted_tours = sorted(result_with_probabilities, key=lambda x: x[2])

    # Select the top 10% highest probability tours
    top_tours = sorted_tours[-num_top_tours:]  # Take from the end

    return top_tours

def add_visited_cells_count(tour_pool):
    """
    Adds the number of visited cells to each tour tuple.
    Returns a list of (tour_df, move_sequence, probability_sum, nb_cells_in_tour).
    """
    result_with_counts = []

    for tour_df, move_sequence, probability_sum in tour_pool:
        nb_cells_in_tour = len(tour_df)  # Count the number of rows (visited cells)
        result_with_counts.append((tour_df, move_sequence, probability_sum, nb_cells_in_tour))
    
    return result_with_counts

def filter_tours_by_cell_count(tour_pool_with_counts, cell_diff):
    """
    Selects tours that have a number of visited cells within ±cell_diff (integer) of the highest probability tour.
    Returns a filtered list of (tour_df, move_sequence, probability_sum, nb_cells_in_tour).
    """
    if not tour_pool_with_counts:
        return []

    # Find the maximum fire probability tour and its number of cells
    max_prob_tour = max(tour_pool_with_counts, key=lambda x: x[2])
    target_cell_count = max_prob_tour[3]  # Number of visited cells in the max probability tour

    # Filter tours within ±3 cells of the target count
    filtered_tours = [
        tour for tour in tour_pool_with_counts
        if target_cell_count - cell_diff <= tour[3] <= target_cell_count + cell_diff
    ]

    return filtered_tours

#We aplly the filters decribed in the technical report in order to generate a good list of candidates
def candidates_generation(start_cell,df,max_moves,pool_size,d_max, max_tours_vertices):
    pool = candidates_pool(start_cell,df,max_moves,pool_size)
    pool_length_corrected = candidate_length_verification(start_cell,pool,d_max)
    pool_big_tours_only =  small_tour_elimination(pool_length_corrected)
    selection = candidate_selection(pool_big_tours_only, max_tours_vertices)
    candidates = filter_tours_by_cell_count(add_visited_cells_count(selection),3)
    return candidates