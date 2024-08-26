from rob_agi.colored_grid import ColoredGrid

def solve_d492a647(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Determines the fill color based on the lowest-numbered non-black, non-gray color found in the input grid.
    2. Creates a deep copy of the input grid.
    3. Applies a checkerboard pattern to black cells:
       - Fills black cells with the determined color where the sum of row and column indices is odd.
       - Preserves all gray (5) cells and non-black, non-gray colors from the input.
       - Leaves other black cells unchanged.
    4. Returns the modified grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the checkerboard pattern applied to black cells.
    """
    # Step 1: Determine the fill color
    fill_color = next((cell for row in input_grid.values for cell in row if cell not in [0, 5]), 1)

    # Step 2: Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Apply the checkerboard pattern to black cells
    for row_index, row in enumerate(output_grid.values):
        for col_index, cell in enumerate(row):
            if cell == 0 and (row_index + col_index) % 2 == 1:
                output_grid.values[row_index][col_index] = fill_color

    # Step 4: Return the modified grid
    return output_grid
