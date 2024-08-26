from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import Counter

def solve_95a58926(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying gray lines and scattered colors,
    then reconstructing the grid with a specific pattern.

    1. Analyzes the input grid to identify gray (5) lines and the scattered color.
    2. Determines the intersection color as the minimum of scattered color and gray (5).
    3. Creates a new grid with the transformed pattern:
       - Fills the background with black (0).
       - Draws horizontal and vertical gray lines.
       - Marks intersections with the determined intersection color.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    gray_rows = find_full_color_rows(input_grid, 5)
    gray_cols = find_full_color_cols(input_grid, 5)
    scattered_color = identify_scattered_color(input_grid)
    intersection_color = min(scattered_color, 5)

    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Fill gray lines
    for row in gray_rows:
        new_grid.values[row] = [5] * cols
    for col in gray_cols:
        for row in range(rows):
            new_grid.values[row][col] = 5

    # Mark intersections
    for row in gray_rows:
        for col in gray_cols:
            new_grid.values[row][col] = intersection_color

    return new_grid

def find_full_color_rows(grid: ColoredGrid, color: int) -> Set[int]:
    return {row for row, row_values in enumerate(grid.values) if all(cell == color for cell in row_values)}

def find_full_color_cols(grid: ColoredGrid, color: int) -> Set[int]:
    rows, cols = grid.get_dimensions()
    return {col for col in range(cols) if all(grid.values[row][col] == color for row in range(rows))}

def identify_scattered_color(grid: ColoredGrid) -> int:
    color_counts = Counter(cell for row in grid.values for cell in row if cell not in {0, 5})
    return max(color_counts, key=color_counts.get) if color_counts else 0
