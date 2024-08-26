from rob_agi.colored_grid import ColoredGrid

def solve_ed98d772(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by:
    1. Copying the input to the top-left quadrant.
    2. Mirroring the top-left quadrant horizontally to create the top-right quadrant.
    3. Mirroring the entire top half vertically to create the bottom half.
    """
    # Create the initial 6x6 output grid
    output_grid = [[0 for _ in range(6)] for _ in range(6)]

    # Copy input to top-left quadrant
    for r in range(3):
        for c in range(3):
            output_grid[r][c] = input_grid.values[r][c]

    # Mirror top-left to top-right
    for r in range(3):
        for c in range(3):
            output_grid[r][5-c] = output_grid[r][c]

    # Mirror top half to bottom half
    for r in range(3):
        for c in range(6):
            output_grid[5-r][c] = output_grid[r][c]

    return ColoredGrid(values=output_grid)
