from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from itertools import cycle

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving vertical color lines from the left side
    to fill a sky blue (8) region on the right side of the grid.

    1. Identifies unique vertical color lines on the left side of the grid.
    2. Creates a cyclic color sequence starting with the second color (or first if only one).
    3. Replaces sky blue (8) cells with colors from the sequence.
    4. Clears the left side of the grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Identify unique vertical color lines
    colors = identify_vertical_lines(input_grid)

    # Step 2: Create cyclic color sequence
    color_cycle = cycle(colors[1:] if len(colors) > 1 else colors)

    # Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Replace sky blue cells with colors from the sequence
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if output_grid.values[r][c] == 8:
                output_grid.values[r][c] = next(color_cycle)

    # Step 4: Clear the left side of the grid
    for r in range(output_grid.num_rows):
        for c in range(7):  # Clear first 7 columns
            output_grid.values[r][c] = 0

    return output_grid

def identify_vertical_lines(grid: ColoredGrid) -> List[int]:
    colors = []
    for col in range(7):  # Assume vertical lines are within first 7 columns
        color = next((cell for cell in grid.values[col] if cell != 0), None)
        if color and color not in colors:
            colors.append(color)
    return colors
