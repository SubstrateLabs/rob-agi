from rob_agi.colored_grid import ColoredGrid

def solve_6ea4a07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing the non-zero color with black (0) and
    the black cells with a new color calculated based on the input color.
    The new color is determined by the formula: new_color = (10 - input_color) % 5,
    then if new_color is 0, it becomes 5.
    """
    # Get dimensions
    rows, cols = input_grid.get_dimensions()
    
    # Find non-zero color
    input_color = next(cell for row in input_grid.values for cell in row if cell != 0)
    
    # Calculate new color
    new_color = (10 - input_color) % 5
    new_color = 5 if new_color == 0 else new_color
    
    # Create new grid
    new_grid = [
        [0 if cell == input_color else new_color for cell in row]
        for row in input_grid.values
    ]
    
    # Return new ColoredGrid
    return ColoredGrid(values=new_grid)
