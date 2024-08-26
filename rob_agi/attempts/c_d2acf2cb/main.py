from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from itertools import combinations

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For pairs of yellow squares in the same row:
       - Creates a path of alternating sky blue (8) and orange (7) between them.
    3. Preserves existing structures and yellow squares.
    4. Ensures no orange or sky blue squares are touching horizontally or vertically.
    5. Respects existing patterns and structures when creating paths.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find all yellow squares
    yellow_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    # Process pairs of yellow squares in the same row
    for (r1, c1), (r2, c2) in combinations(yellow_squares, 2):
        if r1 == r2:  # Same row
            start, end = min(c1, c2), max(c1, c2)
            color = 8  # Start with sky blue
            for c in range(start + 1, end):
                if output_grid.values[r1][c] == 6:  # Respect existing magenta
                    continue
                output_grid.values[r1][c] = color
                color = 7 if color == 8 else 8  # Alternate colors

    # Clean up and resolve adjacencies
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                if any(output_grid.values[nr][nc] in [7, 8] for nr, nc in neighbors):
                    output_grid.values[r][c] = 6  # Revert to magenta if adjacent

    return output_grid
