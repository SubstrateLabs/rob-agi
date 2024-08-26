from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Identifies cell boundaries defined by sky blue (8) lines.
    2. Expands red (2) squares to fill their entire cell.
    3. Expands green (3) squares horizontally to fill their cell and adjacent cells in the same row.
    4. Processes colors in order: green, then red.
    5. Respects sky blue (8) grid lines as boundaries.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find cell boundaries
    cell_boundaries = find_cell_boundaries(output_grid)

    # Process green squares first
    green_squares = find_colored_squares(output_grid, 3)
    for square in green_squares:
        expand_green(output_grid, square, cell_boundaries)

    # Process red squares
    red_squares = find_colored_squares(output_grid, 2)
    for square in red_squares:
        expand_red(output_grid, square, cell_boundaries)

    return output_grid

def find_cell_boundaries(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """Finds the boundaries of cells defined by sky blue (8) lines."""
    rows, cols = grid.get_dimensions()
    boundaries = []
    start_row, start_col = 0, 0

    for r in range(rows):
        if all(grid.values[r][c] == 8 for c in range(cols)):
            if start_row < r:
                for c in range(cols):
                    if grid.values[start_row][c] == 8:
                        if start_col < c:
                            boundaries.append((start_row, start_col, r - 1, c - 1))
                        start_col = c + 1
            start_row = r + 1
            start_col = 0

    return boundaries

def find_colored_squares(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    """Finds all squares of a specific color in the grid."""
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == color]

def expand_green(grid: ColoredGrid, square: Tuple[int, int], boundaries: List[Tuple[int, int, int, int]]):
    """Expands green squares horizontally within their cell and to adjacent cells."""
    row, col = square
    cell = next((b for b in boundaries if b[0] <= row <= b[2] and b[1] <= col <= b[3]), None)
    if cell:
        top, left, bottom, right = cell
        for c in range(left, right + 1):
            if all(grid.values[r][c] in [0, 3] for r in range(top, bottom + 1)):
                for r in range(top, bottom + 1):
                    grid.values[r][c] = 3

def expand_red(grid: ColoredGrid, square: Tuple[int, int], boundaries: List[Tuple[int, int, int, int]]):
    """Expands red squares to fill their entire cell."""
    row, col = square
    cell = next((b for b in boundaries if b[0] <= row <= b[2] and b[1] <= col <= b[3]), None)
    if cell:
        top, left, bottom, right = cell
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 2
