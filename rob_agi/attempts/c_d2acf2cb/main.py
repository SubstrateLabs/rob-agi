from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from itertools import combinations

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For pairs of yellow squares in the same row or column:
       - Removes orange (7) and sky blue (8) squares between them.
       - Replaces removed squares with magenta (6) if part of a magenta structure, otherwise with black (0).
    3. Preserves existing structures, yellow squares, and other colors.
    4. Maintains overall grid structure and symmetry.

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
            start, end = min(c1, c2), max(c1, c2)
            for c in range(start + 1, end):
                if output_grid.values[r1][c] in [7, 8]:
                    # Replace with magenta if adjacent to magenta, otherwise black
                    if any(output_grid.values[r1][cc] == 6 for cc in [c-1, c+1] if 0 <= cc < cols):
                        output_grid.values[r1][c] = 6
                    else:
                        output_grid.values[r1][c] = 0
        elif c1 == c2:  # Same column
            start, end = min(r1, r2), max(r1, r2)
            for r in range(start + 1, end):
                if output_grid.values[r][c1] in [7, 8]:
                    # Replace with magenta if adjacent to magenta, otherwise black
                    if any(output_grid.values[rr][c1] == 6 for rr in [r-1, r+1] if 0 <= rr < rows):
                        output_grid.values[r][c1] = 6
                    else:
                        output_grid.values[r][c1] = 0

    return output_grid
