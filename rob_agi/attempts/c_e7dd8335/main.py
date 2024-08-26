from rob_agi.colored_grid import ColoredGrid

def solve_e7dd8335(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing the bottom half of blue (1) areas to red (2).
    The transformation starts from the vertical midpoint of the grid (inclusive).
    All other colors remain unchanged.
    """
    # Get the dimensions of the input grid
    height, width = input_grid.get_dimensions()

    # Calculate the midpoint (including the middle row in the bottom half)
    midpoint = (height - 1) // 2

    # Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()

    # Iterate through the bottom half of the grid (including the midpoint)
    for i in range(midpoint, height):
        for j in range(width):
            if new_grid.values[i][j] == 1:  # If the cell is blue
                new_grid.values[i][j] = 2  # Change it to red

    return new_grid
