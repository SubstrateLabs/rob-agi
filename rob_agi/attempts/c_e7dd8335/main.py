from rob_agi.colored_grid import ColoredGrid

def solve_e7dd8335(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing the bottom half of blue (1) areas to red (2).
    The transformation occurs at the vertical midpoint of the grid.
    All other colors remain unchanged.
    """
    # Calculate the midpoint
    height, width = input_grid.get_dimensions()
    midpoint = height // 2

    # Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()

    # Iterate through the new grid
    for i in range(height):
        for j in range(width):
            if i >= midpoint and new_grid.values[i][j] == 1:
                new_grid.values[i][j] = 2

    return new_grid
