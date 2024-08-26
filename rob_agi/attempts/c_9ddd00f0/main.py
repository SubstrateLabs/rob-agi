from rob_agi.colored_grid import ColoredGrid

def solve_9ddd00f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by extending patterns both horizontally and vertically.
    
    The function processes the grid as follows:
    1. Keeps the bottom-right quadrant of the grid unchanged.
    2. Mirrors the bottom-right quadrant horizontally to fill the bottom-left quadrant.
    3. Mirrors the entire bottom half vertically to fill the top half of the grid.
    
    This creates a pattern where both horizontal and vertical symmetry is achieved,
    with the bottom-right quadrant serving as the source for the entire grid's pattern.
    """
    height, width = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    mid_row, mid_col = height // 2, width // 2

    # Step 1: Mirror bottom-right to bottom-left
    for row in range(mid_row, height):
        for col in range(mid_col):
            output_grid.values[row][col] = input_grid.values[row][width - 1 - col]

    # Step 2: Mirror bottom half to top half
    for row in range(mid_row):
        output_grid.values[row] = output_grid.values[height - 1 - row].copy()

    return output_grid
