from rob_agi.colored_grid import ColoredGrid

def solve_6fa7a44f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 6fa7a44f challenge by doubling the height of the input grid
    and creating a vertical mirror image in the bottom half.
    
    The function takes an input grid and returns a new grid with twice the height,
    where the top half is the original input and the bottom half is its vertical mirror image.
    """
    # Get the dimensions of the input grid
    height, width = input_grid.get_dimensions()
    
    # Create a new grid with twice the height of the input
    output_values = input_grid.values + input_grid.values[::-1]
    
    # Create and return the new ColoredGrid
    return ColoredGrid(values=output_values)
