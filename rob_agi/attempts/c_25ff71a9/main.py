from rob_agi.colored_grid import ColoredGrid

def solve_25ff71a9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero elements down by one row.
    The top row always becomes all zeros.
    If there are non-zero elements in the bottom row, they wrap around to the second row from the top.
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Move all rows down by one, wrapping the bottom row to the second row
    for row in range(height):
        for col in range(width):
            if row == 1:
                # Second row from top gets values from the bottom row
                output.set_cell(row, col, input_grid.get_cell(height - 1, col))
            elif row > 1:
                # All other rows get values from the row above in the input grid
                output.set_cell(row, col, input_grid.get_cell(row - 1, col))
    
    return output
