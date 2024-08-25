from rob_agi.colored_grid import ColoredGrid

def solve_69889d6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing a diagonal red line from bottom-left to top-right,
    starting from the position of the first red square found. The line is 2 cells wide and
    maintains this width until it reaches the top or right edge. Non-black cells from the
    input are preserved.

    1. Find the starting column of the red square.
    2. Initialize a new grid filled with black.
    3. Draw the diagonal red line, maintaining 2-cell width.
    4. Preserve non-black cells from the input.
    5. Return the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the starting column
    start_col = None
    for row in range(rows - 1, -1, -1):
        for col in range(cols):
            if input_grid.get_cell(row, col) == 2:
                start_col = col
                break
        if start_col is not None:
            break
    
    if start_col is None:
        return input_grid  # No red square found, return the input grid unchanged

    # Initialize the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Draw the diagonal red line
    current_col = start_col
    for row in range(rows - 1, -1, -1):
        output_grid.set_cell(row, current_col, 2)
        if current_col + 1 < cols:
            output_grid.set_cell(row, current_col + 1, 2)
        
        if current_col < cols - 1:
            current_col += 1

    # Preserve non-black cells from the input
    for row in range(rows):
        for col in range(cols):
            if input_grid.get_cell(row, col) != 0:
                output_grid.set_cell(row, col, input_grid.get_cell(row, col))

    return output_grid
