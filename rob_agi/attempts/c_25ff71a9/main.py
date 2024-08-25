from rob_agi.colored_grid import ColoredGrid

def solve_25ff71a9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero elements down by one row,
    setting the top row to all zeros.
    If there's only one row with non-zero elements, move it to the second row from the top.
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Find the non-zero rows
    non_zero_rows = [i for i in range(height) if any(input_grid.get_cell(i, j) != 0 for j in range(width))]
    
    if not non_zero_rows:
        # If all rows are zero, return the input grid as is
        return input_grid
    elif len(non_zero_rows) == 1:
        # If there's only one non-zero row, move it to the second row from the top
        for col in range(width):
            output.set_cell(1, col, input_grid.get_cell(non_zero_rows[0], col))
    else:
        # Move each row down by one
        for row in range(height - 1, 0, -1):
            for col in range(width):
                output.set_cell(row, col, input_grid.get_cell(row - 1, col))
    
    return output
