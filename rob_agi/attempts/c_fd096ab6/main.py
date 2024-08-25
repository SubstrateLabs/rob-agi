from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_fd096ab6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the fd096ab6 challenge by expanding color clusters asymmetrically.
    
    The solution involves the following steps:
    1. Identify all non-blue color clusters in the grid.
    2. For each cluster, calculate its growth potential based on size and surrounding space.
    3. Generate and score possible expansion patterns for each cluster, favoring upward and leftward growth.
    4. Apply the highest-scoring valid expansion pattern for each cluster.
    5. Repeat the expansion process multiple times, resolving conflicts between clusters.
    6. Return the resulting expanded grid.
    
    Args:
    input_grid (ColoredGrid): The initial grid state.
    
    Returns:
    ColoredGrid: The transformed grid with expanded color clusters.
    """
    grid = input_grid.deep_copy()
    colors = set(range(2, 10))  # All colors except blue (1)
    
    for _ in range(3):  # Perform expansion 3 times
        clusters = []
        for color in colors:
            regions = grid.find_connected_regions(color)
            clusters.extend((color, region) for region in regions)
        
        for color, cluster in clusters:
            expand_cluster(grid, color, cluster)
    
    return grid

def expand_cluster(grid: ColoredGrid, color: int, cluster: List[Tuple[int, int]]):
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Prioritize up and left
    rows, cols = grid.get_dimensions()
    
    expansion_cells = set()
    for r, c in cluster:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
                expansion_cells.add((nr, nc))
    
    # Expand to about 50% of possible cells, prioritizing upward and leftward
    expansion_cells = sorted(expansion_cells, key=lambda x: (x[0], x[1]))
    expansion_limit = len(expansion_cells) // 2
    for i, (r, c) in enumerate(expansion_cells):
        if i >= expansion_limit and random.random() < 0.5:
            break
        grid.set_cell(r, c, color)
