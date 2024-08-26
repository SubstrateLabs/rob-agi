from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d4b1c2b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expands a colored grid based on the following rules:
    1. If the grid is uniform (all cells have the same color), return the input grid unchanged.
    2. Otherwise, expand each cell into a 3x3 square region, preserving the color of each original cell in its expanded region.
    """
    if is_uniform(input_grid):
        return input_grid

    expansion_factor = 3
    
    rows, cols = input_grid.get_dimensions()
    new_rows, new_cols = rows * expansion_factor, cols * expansion_factor
    output_grid = ColoredGrid(values=[[0 for _ in range(new_cols)] for _ in range(new_rows)])

    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            for dr in range(expansion_factor):
                for dc in range(expansion_factor):
                    output_grid.set_cell(r * expansion_factor + dr, c * expansion_factor + dc, color)

    return output_grid

def is_uniform(grid: ColoredGrid) -> bool:
    """Check if all cells in the grid have the same color."""
    first_color = grid.get_cell(0, 0)
    rows, cols = grid.get_dimensions()
    return all(grid.get_cell(r, c) == first_color for r in range(rows) for c in range(cols))

def find_largest_region_size(grid: ColoredGrid) -> int:
    """Find the size of the largest connected region in the grid."""
    rows, cols = grid.get_dimensions()
    visited = set()
    largest_size = 0

    def dfs(r: int, c: int, color: int) -> int:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.get_cell(r, c) != color:
            return 0
        visited.add((r, c))
        size = 1
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            size += dfs(r + dr, c + dc, color)
        return size

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                largest_size = max(largest_size, dfs(r, c, grid.get_cell(r, c)))

    return largest_size

def get_expansion_factor(region_size: int) -> int:
    """Determine the expansion factor based on the largest region size."""
    return 3  # Always return 3 for non-uniform grids
