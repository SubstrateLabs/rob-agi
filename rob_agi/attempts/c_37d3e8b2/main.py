from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def partition_grid(dimensions: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    rows, cols = dimensions
    regions = []
    for i in range(0, rows, 5):
        for j in range(0, cols, 5):
            regions.append((i, j, min(i+5, rows), min(j+5, cols)))
    return regions

def has_sky_blue_in_region(grid: List[List[int]], region: Tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = region
    return any(grid[r][c] == 8 for r in range(top, bottom) for c in range(left, right))

def color_region(grid: List[List[int]], region: Tuple[int, int, int, int], color: int) -> None:
    top, left, bottom, right = region
    for r in range(top, bottom):
        for c in range(left, right):
            if grid[r][c] == 8:
                grid[r][c] = color

def solve_37d3e8b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by partitioning the grid into regions
    and coloring sky blue (8) cells in each region based on a specific color sequence.
    
    The solution follows these steps:
    1. Create a deep copy of the input grid.
    2. Partition the grid into 5x5 regions (or smaller at edges).
    3. Initialize the color sequence [1, 2, 3, 7].
    4. For each region:
       a. Check if the region contains any sky blue (8) cells.
       b. If it does, color all sky blue cells in the region with the current color.
       c. Move to the next color in the sequence.
    5. Return the transformed grid.
    """
    grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 7]
    color_index = 0
    
    regions = partition_grid(grid.get_dimensions())
    
    for region in regions:
        if has_sky_blue_in_region(grid.values, region):
            current_color = color_sequence[color_index]
            color_region(grid.values, region, current_color)
            color_index = (color_index + 1) % len(color_sequence)
    
    return grid
