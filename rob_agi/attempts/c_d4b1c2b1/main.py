from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_d4b1c2b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expands a colored grid based on the following rules:
    1. If the grid is uniform (all cells have the same color), return the input grid unchanged.
    2. Otherwise, expand each cell into a square region, where the size of the square is determined by the number of unique colors in the input grid.
    """
    if is_uniform(input_grid):
        return input_grid

    expansion_factor = count_unique_colors(input_grid)
    expanded_grid = create_expanded_grid(input_grid, expansion_factor)
    fill_expanded_grid(input_grid, expanded_grid, expansion_factor)

    return expanded_grid

def is_uniform(grid: ColoredGrid) -> bool:
    """Check if all cells in the grid have the same color."""
    first_color = grid.get_cell(0, 0)
    rows, cols = grid.get_dimensions()
    return all(grid.get_cell(r, c) == first_color for r in range(rows) for c in range(cols))

def count_unique_colors(grid: ColoredGrid) -> int:
    """Count the number of unique colors in the grid."""
    unique_colors = set()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            unique_colors.add(grid.get_cell(r, c))
    return len(unique_colors)

def create_expanded_grid(original_grid: ColoredGrid, expansion_factor: int) -> ColoredGrid:
    """Create an expanded grid based on the original grid and expansion factor."""
    rows, cols = original_grid.get_dimensions()
    new_rows, new_cols = rows * expansion_factor, cols * expansion_factor
    return ColoredGrid(values=[[0 for _ in range(new_cols)] for _ in range(new_rows)])

def fill_expanded_grid(original_grid: ColoredGrid, expanded_grid: ColoredGrid, expansion_factor: int) -> None:
    """Fill the expanded grid with colors from the original grid."""
    rows, cols = original_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = original_grid.get_cell(r, c)
            for dr in range(expansion_factor):
                for dc in range(expansion_factor):
                    expanded_grid.set_cell(r * expansion_factor + dr, c * expansion_factor + dc, color)
