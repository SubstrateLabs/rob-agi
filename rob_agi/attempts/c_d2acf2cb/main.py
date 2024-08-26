from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from itertools import combinations

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For pairs of yellow squares in the same row or column:
       - Replaces orange (7) and sky blue (8) squares between them with magenta (6) or black (0).
       - If the square to be replaced is adjacent to an existing magenta square, it becomes magenta.
       - Otherwise, it becomes black.
    3. Preserves existing structures, yellow squares, and other colors.
    4. Maintains overall grid structure and symmetry.
    5. Special case: In rows with yellow squares at both ends, replaces all squares between with alternating
       magenta and black, starting with magenta next to the yellow.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find all yellow squares
    yellow_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    # Process pairs of yellow squares in the same row or column
    for (r1, c1), (r2, c2) in combinations(yellow_squares, 2):
        if r1 == r2:  # Same row
            process_line(output_grid, r1, c1, c2, horizontal=True)
        elif c1 == c2:  # Same column
            process_line(output_grid, c1, r1, r2, horizontal=False)

    return output_grid

def process_line(grid: ColoredGrid, fixed: int, start: int, end: int, horizontal: bool):
    start, end = min(start, end), max(start, end)
    for i in range(start + 1, end):
        r, c = (fixed, i) if horizontal else (i, fixed)
        if grid.values[r][c] in [7, 8]:
            if has_adjacent_magenta(grid, r, c):
                grid.values[r][c] = 6
            else:
                grid.values[r][c] = 0
    
    # Special case for rows with yellow at both ends
    if horizontal and grid.values[fixed][start] == 4 and grid.values[fixed][end] == 4:
        for i in range(start + 1, end):
            grid.values[fixed][i] = 6 if (i - start) % 2 == 1 else 0

def has_adjacent_magenta(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 6:
            return True
    return False
