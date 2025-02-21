# --- Levenshtein Distance for String Kernel ---
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# --- Chamfer Distance for Point Cloud Kernel ---
def chamfer_distance(R: List[Tuple[float, float]], R_prime: List[Tuple[float, float]]) -> float:
    """Compute Chamfer distance between two point clouds."""

    def nearest_neighbor_dist(point, points):
        return min(np.linalg.norm(np.array(point) - np.array(p)) for p in points) if points else 0

    d1 = sum(nearest_neighbor_dist(r, R_prime) for r in R) / (len(R) or 1)
    d2 = sum(nearest_neighbor_dist(r_prime, R) for r_prime in R_prime) / (len(R_prime) or 1)
    return (d1 + d2) / 2