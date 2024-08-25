from rob_agi.colored_grid import ColoredGrid

def solve_a79310a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by shifting all 8s one row down and changing them to 2s.
    If an 8 is in the last row, it disappears in the output.
    """
    height, width = input_grid.get_dimensions()
    output = input_grid.deep_copy()
    
    # Clear the output grid
    for row in range(height):
        for col in range(width):
            output.set_cell(row, col, 0)
    
    # Apply the transformation
    for row in range(height - 1):  # Exclude the last row
        for col in range(width):
            if input_grid.get_cell(row, col) == 8:
                output.set_cell(row + 1, col, 2)
    
    return output
