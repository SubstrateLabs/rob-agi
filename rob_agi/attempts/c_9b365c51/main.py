from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by projecting vertical color lines from the left side
    onto sky blue (8) regions on the right side of the grid.

    1. Identifies the sequence of unique colors from the left side of the grid.
    2. Creates a deep copy of the input grid.
    3. Clears the left side of the grid (first 7 columns).
    4. Replaces sky blue (8) cells and their vertical sections on the right side 
       with colors from the identified sequence, using a new color for each 
       contiguous vertical section containing sky blue cells.
    5. Returns the transformed grid.

    The color sequence wraps around if more colors are needed than are present
    in the original sequence.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Identify the color sequence and find the leftmost sky blue column
    color_sequence, leftmost_sky_blue_col = analyze_input_grid(input_grid)

    # Step 2: Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Clear the left side of the grid
    for row in range(output_grid.num_rows):
        for col in range(7):
            output_grid.values[row][col] = 0

    # Step 4: Process the right side of the grid
    color_index = 0
    for col in range(leftmost_sky_blue_col, output_grid.num_cols):
        if any(output_grid.values[row][col] == 8 for row in range(output_grid.num_rows)):
            current_color = color_sequence[color_index % len(color_sequence)]
            for row in range(output_grid.num_rows):
                if output_grid.values[row][col] == 8:
                    output_grid.values[row][col] = current_color
            color_index += 1

    # Step 5: Return the transformed grid
    return output_grid

def analyze_input_grid(grid: ColoredGrid) -> Tuple[List[int], int]:
    colors = []
    leftmost_sky_blue_col = grid.num_cols
    for col in range(7):
        for row in range(grid.num_rows):
            color = grid.values[row][col]
            if color != 0 and color not in colors:
                colors.append(color)
    for col in range(7, grid.num_cols):
        if any(grid.values[row][col] == 8 for row in range(grid.num_rows)):
            leftmost_sky_blue_col = col
            break
    return colors, leftmost_sky_blue_col
