from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import numpy as np

def solve_de493100(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the de493100 challenge by finding an interesting subgrid within the input grid.
    
    The solution works by:
    1. Iterating through all possible subgrids of sizes 4x4 to 10x10
    2. Scoring each subgrid based on color diversity, continuity, and uniqueness
    3. Selecting the subgrid with the highest score
    
    The scoring mechanism aims to balance:
    - Diversity of colors (more unique colors are preferred)
    - Continuity of colors (some level of pattern is preferred)
    - Uniqueness compared to the full grid (subgrids that stand out are preferred)
    
    Args:
    input_grid (ColoredGrid): The input 30x30 grid

    Returns:
    ColoredGrid: The most interesting subgrid found
    """
    best_subgrid = None
    best_score = -float('inf')
    min_size, max_size = 4, 10

    for r in range(30):
        for c in range(30):
            for size in range(min_size, max_size + 1):
                if r + size <= 30 and c + size <= 30:
                    subgrid = input_grid.extract_subgrid(r, c, size, size)
                    score = score_subgrid(subgrid, input_grid)
                    if score > best_score:
                        best_score = score
                        best_subgrid = subgrid

    return best_subgrid

def score_subgrid(subgrid: ColoredGrid, full_grid: ColoredGrid) -> float:
    diversity_score = len(set(color for row in subgrid.values for color in row))
    continuity_score = calculate_continuity(subgrid)
    uniqueness_score = calculate_uniqueness(subgrid, full_grid)
    
    return diversity_score * 0.4 + continuity_score * 0.3 + uniqueness_score * 0.3

def calculate_continuity(grid: ColoredGrid) -> float:
    continuity = 0
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if r > 0 and grid.values[r][c] == grid.values[r-1][c]:
                continuity += 1
            if c > 0 and grid.values[r][c] == grid.values[r][c-1]:
                continuity += 1
    return continuity / (2 * len(grid.values) * len(grid.values[0]) - len(grid.values) - len(grid.values[0]))

def calculate_uniqueness(subgrid: ColoredGrid, full_grid: ColoredGrid) -> float:
    subgrid_colors = [color for row in subgrid.values for color in row]
    full_grid_colors = [color for row in full_grid.values for color in row]
    
    subgrid_dist = np.histogram(subgrid_colors, bins=range(11))[0]
    full_grid_dist = np.histogram(full_grid_colors, bins=range(11))[0]
    
    subgrid_dist = subgrid_dist / np.sum(subgrid_dist)
    full_grid_dist = full_grid_dist / np.sum(full_grid_dist)
    
    return np.sum(np.abs(subgrid_dist - full_grid_dist))
