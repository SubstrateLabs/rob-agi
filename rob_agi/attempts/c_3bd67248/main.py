from rob_agi.colored_grid import ColoredGrid

def solve_3bd67248(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid according to the following pattern:
    1. Keep the leftmost column unchanged
    2. Add a diagonal line of 2's from top-right to bottom-left
    3. Fill the bottom row with 4's, except for the leftmost cell
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    for i in range(height):
        # Copy leftmost column
        output.set_cell(i, 0, input_grid.get_cell(i, 0))
        
        # Add diagonal 2's
        if i < width - 1:
            output.set_cell(i, width - 1 - i, 2)
    
    # Fill bottom row with 4's
    for j in range(1, width):
        output.set_cell(height - 1, j, 4)
    
    return output
