from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For rows with yellow squares at both ends:
       - Replaces squares between yellows with alternating orange (7) and sky blue (8).
    3. Preserves existing structures and yellow squares.
    4. Ensures no orange or sky blue squares are touching horizontally or vertically.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find rows with yellow squares at both ends
    for r in range(rows):
        row_yellows = [c for c in range(cols) if output_grid.values[r][c] == 4]
        if len(row_yellows) == 2:
            start, end = row_yellows
            for c in range(start + 1, end):
                output_grid.values[r][c] = 7 if (c - start) % 2 else 8

    # Adjust for color adjacency
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                if any(output_grid.values[nr][nc] in [7, 8] for nr, nc in neighbors):
                    output_grid.values[r][c] = 6

    return output_grid
