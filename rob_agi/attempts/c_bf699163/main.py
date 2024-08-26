from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_bf699163(input_grid: ColoredGrid) -> Optional[ColoredGrid]:
    """
    Solves the bf699163 challenge by finding the most central valid 3x3 pattern in the input grid.

    A valid pattern is a 3x3 subgrid with a gray (5) center and all surrounding cells
    of the same non-gray color. The most central pattern is determined by its proximity
    to the center of the entire grid. If multiple patterns are equally central,
    the one with the lowest color value is chosen.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A 3x3 grid representing the most central valid pattern,
                 or None if no valid pattern is found.
    """
    def is_valid_pattern(grid: ColoredGrid, row: int, col: int) -> bool:
        if grid.values[row][col] != 5:
            return False
        color = grid.values[row-1][col]
        if color == 5:
            return False
        return all(
            grid.values[r][c] == color
            for r in range(row-1, row+2)
            for c in range(col-1, col+2)
            if (r, c) != (row, col)
        )

    def calculate_centrality(row: int, col: int, center_row: float, center_col: float) -> float:
        return abs(row - center_row) + abs(col - center_col)

    rows, cols = input_grid.get_dimensions()
    if rows < 3 or cols < 3:
        return None

    center_row = (rows - 1) / 2
    center_col = (cols - 1) / 2

    valid_patterns = []
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if is_valid_pattern(input_grid, row, col):
                color = input_grid.values[row-1][col]
                centrality = calculate_centrality(row, col, center_row, center_col)
                valid_patterns.append((row, col, color, centrality))

    if not valid_patterns:
        return None

    most_central = min(valid_patterns, key=lambda x: (x[3], x[2]))
    color = most_central[2]

    return ColoredGrid(values=[
        [color, color, color],
        [color, 5, color],
        [color, color, color]
    ])
