from rob_agi.colored_grid import ColoredGrid

def solve_db3e9e38(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating an inverted triangle pattern centered on a vertical line of 7s.
    
    The function:
    1. Identifies the central column containing the vertical line of 7s.
    2. Determines the last row where a 7 appears in this central column.
    3. Creates an inverted triangle pattern:
       - Starts from the top of the grid and works downwards.
       - For the first two rows, fills the entire width with alternating 8s and 7s, starting with 8.
       - For subsequent rows, maintains the alternating pattern but reduces the width by one cell on each side.
       - Centers the pattern on the column of 7s.
       - Stops the pattern at the last row where a 7 appears in the central column.
    4. Ensures that the central column always contains 7s.
    5. Fills all other cells with 0s.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the inverted triangle pattern.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    # Find the central column with 7s
    central_col = next((col for col in range(width) if grid.get_cell(0, col) == 7), None)

    if central_col is None:
        return grid  # No 7s found, return original grid

    # Find the last row with a 7 in the central column
    last_seven_row = next((row for row in range(height-1, -1, -1) if grid.get_cell(row, central_col) == 7), -1)

    # Create the inverted triangle pattern
    for row in range(height):
        if row > last_seven_row:
            break

        pattern_width = min(width, last_seven_row - row + 1)
        start_col = max(0, central_col - pattern_width + 1)
        end_col = min(width, central_col + pattern_width)

        for col in range(width):
            if col < start_col or col >= end_col:
                grid.set_cell(row, col, 0)
            elif col == central_col:
                grid.set_cell(row, col, 7)
            else:
                offset = abs(col - central_col)
                grid.set_cell(row, col, 8 if offset % 2 == 1 else 7)

    # Clear any remaining cells below the pattern
    for row in range(last_seven_row + 1, height):
        for col in range(width):
            grid.set_cell(row, col, 0)

    return grid
