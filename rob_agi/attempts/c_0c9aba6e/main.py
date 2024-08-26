from rob_agi.colored_grid import ColoredGrid

def solve_0c9aba6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 13x4 input grid into a 6x4 output grid based on the following rules:
    1. For each cell in the output grid:
       - Check the corresponding cell and the one below it in the input grid.
       - If exactly one of these two cells is red (2), set the output cell to sky blue (8).
       - Otherwise, set the output cell to black (0).
    2. Only the first 7 rows of the input grid are considered.
    3. Returns the resulting 6x4 grid.
    """
    # Create a new 6x4 ColoredGrid for the output, initially filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(6)])

    # Iterate through each cell in the output grid
    for r in range(6):
        for c in range(4):
            # Check the corresponding cell and the one below it in the input grid
            current_cell = input_grid.values[r][c]
            cell_below = input_grid.values[r+1][c]

            # Apply the transformation rule
            if (current_cell == 2) != (cell_below == 2):  # XOR operation
                output_grid.values[r][c] = 8  # sky blue
            # If both are 2 or neither is 2, it remains 0 (black)

    return output_grid
