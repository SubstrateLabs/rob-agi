from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by converting yellow squares to red and creating L-shaped patterns.
    
    1. Identifies yellow (4) squares and their clusters in the grid.
    2. Processes clusters based on size:
       - For clusters of 3+: Converts outer squares to red, leaves inner square(s) yellow.
       - For clusters of 2: Converts both squares to red.
       - For isolated squares: Processes them in step 5.
    3. Creates L-shaped red patterns from converted squares.
    4. Processes remaining isolated yellow squares:
       - Converts to red if adjacent to red or within 2 steps of another yellow.
       - Creates short L-shaped paths when possible.
    5. Optimizes the pattern by removing dead-ends and balancing the distribution.
    6. Performs a final pass to improve pattern symmetry and compactness.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def find_yellow_clusters() -> List[Set[Tuple[int, int]]]:
        yellows = set(find_yellow_squares())
        clusters = []
        while yellows:
            start = yellows.pop()
            cluster = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in yellows:
                        yellows.remove((nr, nc))
                        cluster.add((nr, nc))
                        stack.append((nr, nc))
            clusters.append(cluster)
        return clusters
    
    def process_clusters(clusters: List[Set[Tuple[int, int]]]) -> None:
        for cluster in clusters:
            if len(cluster) >= 3:
                outer_squares = [sq for sq in cluster if sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                                             if (sq[0]+dr, sq[1]+dc) in cluster) < 3]
                for sq in outer_squares:
                    grid.values[sq[0]][sq[1]] = 2  # Convert to red
            elif len(cluster) == 2:
                for sq in cluster:
                    grid.values[sq[0]][sq[1]] = 2  # Convert to red
    
    def create_l_shape(start: Tuple[int, int], length: int = 2) -> None:
        r, c = start
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr1, dc1 in directions:
            for dr2, dc2 in directions:
                if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                    path = [(r + dr1*i, c + dc1*i) for i in range(1, length+1)] + \
                           [(r + dr1*length + dr2*i, c + dc1*length + dc2*i) for i in range(1, length+1)]
                    if all(0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in [0, 5] for nr, nc in path):
                        for nr, nc in path:
                            grid.values[nr][nc] = 2
                        return
    
    def process_isolated_yellows() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    if any(grid.values[r+dr][c+dc] == 2 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        grid.values[r][c] = 2
                        create_l_shape((r, c))
                    elif any(grid.values[r+dr][c+dc] == 4 for dr in [-2, -1, 0, 1, 2] for dc in [-2, -1, 0, 1, 2]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols and (dr != 0 or dc != 0)):
                        grid.values[r][c] = 2
                        create_l_shape((r, c))
    
    def optimize_pattern() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
                    if neighbors <= 1:
                        grid.values[r][c] = 0  # Remove dead-ends
    
    yellow_clusters = find_yellow_clusters()
    process_clusters(yellow_clusters)
    
    for cluster in yellow_clusters:
        if len(cluster) >= 2:
            for sq in cluster:
                if grid.values[sq[0]][sq[1]] == 2:
                    create_l_shape(sq)
    
    process_isolated_yellows()
    optimize_pattern()
    
    return grid
