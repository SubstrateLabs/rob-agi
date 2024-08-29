from rob_agi.colored_grid import ColoredGrid

def solve_25ff71a9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero elements down by one row.
    The top row always becomes all zeros.
    If there are non-zero elements in the bottom row, they wrap around to the second row from the top.
    The relative positions of non-zero elements within each row are preserved.
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    for row in range(height):
        for col in range(width):
            value = input_grid.get_cell(row, col)
            if value != 0:
                if row == height - 1:
                    # Bottom row wraps to second row from top
                    output.set_cell(1, col, value)
                else:
                    # Move down by one row
                    output.set_cell(row + 1, col, value)
    
    return output
