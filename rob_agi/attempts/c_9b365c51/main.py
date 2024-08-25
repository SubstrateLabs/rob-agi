from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving vertical color lines from the left side
    to fill sky blue (8) regions on the right side of the grid.

    1. Identifies unique vertical color lines on the left side of the grid.
    2. Creates a deep copy of the input grid.
    3. Replaces sky blue (8) cells with colors from the identified sequence,
       using a new color for each contiguous sky blue region.
    4. Clears the left side of the grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Identify unique vertical color lines
    colors = identify_vertical_lines(input_grid)

    # Step 2: Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Transform the grid
    current_color_index = 0
    last_8_position = None

    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if c < 7:
                # Clear the left side of the grid
                output_grid.values[r][c] = 0
            elif output_grid.values[r][c] == 8:
                # Replace sky blue cells with colors from the sequence
                if last_8_position is None or (r, c) != (last_8_position[0], last_8_position[1] + 1):
                    current_color_index = (current_color_index + 1) % len(colors)
                output_grid.values[r][c] = colors[current_color_index]
                last_8_position = (r, c)
            else:
                # Reset last_8_position when encountering a non-sky blue cell
                last_8_position = None

    return output_grid

def identify_vertical_lines(grid: ColoredGrid) -> List[int]:
    colors = []
    for col in range(7):  # Assume vertical lines are within first 7 columns
        color = next((cell for cell in grid.values if cell[col] != 0), None)
        if color and color[col] not in colors:
            colors.append(color[col])
    return colors
