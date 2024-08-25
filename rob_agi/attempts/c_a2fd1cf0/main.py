from rob_agi.colored_grid import ColoredGrid

def solve_a2fd1cf0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a2fd1cf0 challenge by creating an 'L'-shaped path of '8's connecting '2' and '3'.
    
    The function finds the positions of '2' and '3' in the input grid, then creates a path:
    1. A horizontal line of '8's from the column of '2' to the column of '3' (exclusive of '2' and '3').
    2. A vertical line of '8's from the row of '2' to the row of '3' (exclusive of '2' and '3').
    
    The path is created regardless of whether '2' is to the left or right of '3', or above or below '3'.
    The '2' and '3' cells are preserved in their original positions.
    If either '2' or '3' is not found, the original grid is returned unchanged.
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    # Find positions of '2' and '3'
    start, end = None, None
    for row in range(height):
        for col in range(width):
            if grid.get_cell(row, col) == 2:
                start = (row, col)
            elif grid.get_cell(row, col) == 3:
                end = (row, col)
        if start and end:
            break

    # If either '2' or '3' is not found, return the original grid
    if not start or not end:
        return grid

    # Create horizontal path
    start_col, end_col = min(start[1], end[1]), max(start[1], end[1])
    for col in range(start_col + 1, end_col):
        grid.set_cell(start[0], col, 8)

    # Create vertical path
    start_row, end_row = min(start[0], end[0]), max(start[0], end[0])
    for row in range(start_row + 1, end_row):
        grid.set_cell(row, end_col if start[1] < end[1] else start_col, 8)

    return grid
