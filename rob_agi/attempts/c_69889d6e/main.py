from rob_agi.colored_grid import ColoredGrid

def solve_69889d6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing a diagonal red line from bottom to top-right,
    starting from the position of the first red square found in the bottom row.
    The line is 2 cells thick, except for the top row, and stops at the top or right edge.
    Non-black cells from the input are preserved.

    1. Find the starting column of the red square in the bottom row.
    2. Initialize a new grid filled with black.
    3. Draw the diagonal red line, maintaining 2-cell thickness.
    4. Preserve non-black cells from the input.
    5. Return the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the starting column in the bottom row
    start_col = None
    for col in range(cols):
        if input_grid.get_cell(rows - 1, col) == 2:
            start_col = col
            break
    
    if start_col is None:
        return input_grid  # No red square found, return the input grid unchanged

    # Initialize the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Draw the diagonal red line
    current_row = rows - 1
    current_col = start_col
    while current_row >= 0 and current_col < cols:
        output_grid.set_cell(current_row, current_col, 2)
        if current_row < rows - 1:
            output_grid.set_cell(current_row + 1, current_col, 2)
        current_row -= 1
        current_col += 1

    # Preserve non-black cells from the input
    for row in range(rows):
        for col in range(cols):
            if input_grid.get_cell(row, col) != 0:
                output_grid.set_cell(row, col, input_grid.get_cell(row, col))

    return output_grid
