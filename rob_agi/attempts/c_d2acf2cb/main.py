from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For rows with yellow squares at both ends:
       - Replaces all squares between them with an alternating pattern of sky (8) and orange (7),
         starting with sky next to the yellow.
    3. Preserves existing structures, yellow squares, and other colors in all other rows.
    4. Maintains overall grid structure and symmetry.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    for r in range(rows):
        yellow_positions = [c for c in range(cols) if output_grid.values[r][c] == 4]
        if len(yellow_positions) == 2:
            start, end = yellow_positions
            for i in range(start + 1, end):
                output_grid.values[r][i] = 8 if (i - start) % 2 == 1 else 7

    return output_grid
