from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from itertools import combinations

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For pairs of yellow squares in the same row or column:
       - Replaces squares between yellows with alternating sky blue (8) and orange (7).
    3. Preserves existing structures and yellow squares.
    4. Ensures no orange or sky blue squares are touching horizontally or vertically.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find all yellow squares
    yellow_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    # Process pairs of yellow squares
    for (r1, c1), (r2, c2) in combinations(yellow_squares, 2):
        if r1 == r2:  # Same row
            start, end = min(c1, c2), max(c1, c2)
            for c in range(start + 1, end):
                output_grid.values[r1][c] = 8 if (c - start) % 2 == 1 else 7
        elif c1 == c2:  # Same column
            start, end = min(r1, r2), max(r1, r2)
            for r in range(start + 1, end):
                output_grid.values[r][c1] = 8 if (r - start) % 2 == 1 else 7

    # Clean up existing patterns and resolve adjacencies
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                if any(output_grid.values[nr][nc] in [7, 8] for nr, nc in neighbors):
                    output_grid.values[r][c] = 6

    return output_grid
