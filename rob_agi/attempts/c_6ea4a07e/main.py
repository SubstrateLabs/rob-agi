from rob_agi.colored_grid import ColoredGrid

def solve_6ea4a07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing the non-zero color with black (0) and
    the black cells with a new color calculated based on the input color.
    The new color is determined by the formula: x = (10 - input_color) % 5,
    then if x is 0, new_color = 4, otherwise new_color = x - 1.
    """
    # Get dimensions
    rows, cols = input_grid.get_dimensions()
    
    # Find non-zero color
    input_color = next(cell for row in input_grid.values for cell in row if cell != 0)
    
    # Calculate new color
    x = (10 - input_color) % 5
    new_color = 4 if x == 0 else x - 1
    
    # Create new grid
    new_grid = [
        [0 if cell == input_color else new_color for cell in row]
        for row in input_grid.values
    ]
    
    # Return new ColoredGrid
    return ColoredGrid(values=new_grid)
