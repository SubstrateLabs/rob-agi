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
    gray_rows = find_gray_rows(input_grid)
    gray_cols = find_gray_cols(input_grid)
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

def find_gray_rows(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [row for row in range(rows) if sum(1 for cell in grid.values[row] if cell == 5) > cols // 2]

def find_gray_cols(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [col for col in range(cols) if sum(1 for row in range(rows) if grid.values[row][col] == 5) > rows // 2]

def identify_scattered_color(grid: ColoredGrid) -> int:
    color_counts = Counter(cell for row in grid.values for cell in row if cell not in {0, 5})
    return max(color_counts, key=color_counts.get) if color_counts else 0
