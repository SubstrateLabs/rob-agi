from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares in the grid.
    2. For each column containing yellow squares:
       - Fills the space between the topmost and bottommost yellow squares with a pattern.
       - The pattern is: sky (8) next to yellow, then orange (7), then sky (8) again if there's space.
    3. The pattern is only applied if the original content was magenta (6) or black (0).
    4. Preserves existing structures, yellow squares, and other colors outside the transformation area.
    5. Maintains overall grid structure and symmetry.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find columns with yellow squares
    yellow_columns = set()
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 4:
                yellow_columns.add(c)

    # Process each column with yellow squares
    for col in yellow_columns:
        yellow_positions = [r for r in range(rows) if output_grid.values[r][col] == 4]
        top, bottom = min(yellow_positions), max(yellow_positions)

        # Fill the column between yellow squares
        for r in range(top + 1, bottom):
            if output_grid.values[r][col] in [0, 6]:
                if r == top + 1 or r == bottom - 1:
                    output_grid.values[r][col] = 8  # sky next to yellow
                else:
                    output_grid.values[r][col] = 7  # orange in between

    return output_grid
