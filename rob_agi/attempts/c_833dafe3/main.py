from rob_agi.colored_grid import ColoredGrid

def solve_833dafe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and creating a symmetrical pattern.
    
    The function creates an output grid twice the size of the input in both dimensions.
    It fills the center with the top-left value of the input, places corner values
    strategically, and creates symmetry by mirroring the input values both horizontally
    and vertically. The result is a symmetrical pattern that preserves certain
    characteristics of the input grid while expanding it.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*cols)] for _ in range(2*rows)])
    
    # Extract corner values
    top_left = input_grid.values[0][0]
    top_right = input_grid.values[0][-1]
    bottom_left = input_grid.values[-1][0]
    bottom_right = input_grid.values[-1][-1]
    
    # Fill the center with top_left value
    for i in range(rows, rows*2):
        for j in range(cols, cols*2):
            new_grid.values[i][j] = top_left
    
    # Place corner values
    new_grid.values[0][0] = bottom_right
    new_grid.values[0][-1] = bottom_right
    new_grid.values[-1][0] = bottom_right
    new_grid.values[-1][-1] = bottom_right
    new_grid.values[0][cols] = top_right
    new_grid.values[-1][cols-1] = top_right
    new_grid.values[rows-1][0] = bottom_left
    new_grid.values[rows][cols-1] = bottom_left
    
    # Fill top-left quarter (excluding center)
    for i in range(rows):
        for j in range(cols):
            if i != rows-1 or j != cols-1:  # Exclude bottom-right cell
                new_grid.values[i][j] = input_grid.values[i][j]
                new_grid.values[i][j+1] = input_grid.values[i][j]
    
    # Mirror top-left quarter horizontally
    for i in range(rows):
        for j in range(cols):
            new_grid.values[i][2*cols-1-j] = new_grid.values[i][j]
    
    # Mirror top half vertically
    for i in range(rows):
        for j in range(2*cols):
            new_grid.values[2*rows-1-i][j] = new_grid.values[i][j]
    
    return new_grid
