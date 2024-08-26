from rob_agi.colored_grid import ColoredGrid

def solve_d492a647(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Determines the fill color based on the lowest-numbered non-black, non-gray color found in the input grid.
       If no such color is found, defaults to blue (1).
    2. Creates a deep copy of the input grid.
    3. Applies a checkerboard pattern to black cells:
       - Fills black cells with the determined color where the sum of row and column indices is odd.
       - Preserves all non-black colors (including gray) from the input.
    4. Returns the modified grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the checkerboard pattern applied to black cells.
    """
    # Step 1: Determine the fill color
    fill_color = find_lowest_non_black_non_gray_color(input_grid)

    # Step 2: Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Apply the checkerboard pattern to black cells
    for row_index, row in enumerate(output_grid.values):
        for col_index, cell in enumerate(row):
            if cell == 0 and (row_index + col_index) % 2 == 1:
                output_grid.values[row_index][col_index] = fill_color

    # Step 4: Return the modified grid
    return output_grid

def find_lowest_non_black_non_gray_color(grid: ColoredGrid) -> int:
    """
    Finds the lowest-numbered non-black, non-gray color in the grid.
    If no such color is found, returns blue (1) as the default.

    Args:
    grid (ColoredGrid): The input grid to search for colors.

    Returns:
    int: The lowest-numbered non-black, non-gray color, or 1 (blue) if none found.
    """
    colors = set(cell for row in grid.values for cell in row if cell not in [0, 5])
    return min(colors) if colors else 1
