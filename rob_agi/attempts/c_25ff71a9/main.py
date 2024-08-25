from rob_agi.colored_grid import ColoredGrid

def solve_25ff71a9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving non-zero elements down by one row,
    keeping the bottom row in place, and setting the top row to all zeros.
    If there's only one row with non-zero elements, move it to the bottom.
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Find the first and last non-zero rows
    first_non_zero_row = next((i for i in range(height) if any(input_grid.get_cell(i, j) != 0 for j in range(width))), -1)
    last_non_zero_row = next((i for i in range(height - 1, -1, -1) if any(input_grid.get_cell(i, j) != 0 for j in range(width))), -1)
    
    if first_non_zero_row == -1:
        # If all rows are zero, return the input grid as is
        return input_grid
    elif first_non_zero_row == last_non_zero_row:
        # If there's only one non-zero row, move it to the bottom
        for col in range(width):
            output.set_cell(height - 1, col, input_grid.get_cell(first_non_zero_row, col))
    else:
        # Move each row down by one, keeping the bottom row in place
        for row in range(height - 1, 0, -1):
            for col in range(width):
                if row == height - 1:
                    # Keep the bottom row of the input in place
                    output.set_cell(row, col, input_grid.get_cell(row, col))
                else:
                    output.set_cell(row, col, input_grid.get_cell(row - 1, col))
    
    return output
