from rob_agi.colored_grid import ColoredGrid

def solve_833dafe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and creating a symmetrical pattern.
    
    The function creates an output grid twice the size of the input in both dimensions.
    It fills the central part by duplicating each input cell into a 2x2 block.
    The edges are then filled to create a symmetrical frame-like structure.
    Special handling is applied to corners and the second/second-to-last rows and columns.
    The result is a symmetrical pattern that expands on the input grid's design while
    maintaining specific rules for the outer frame.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*width)] for _ in range(2*height)])
    
    # Fill the central part
    for i in range(height):
        for j in range(width):
            value = input_grid.values[i][j]
            new_grid.values[2*i][2*j] = new_grid.values[2*i][2*j+1] = value
            new_grid.values[2*i+1][2*j] = new_grid.values[2*i+1][2*j+1] = value
    
    # Handle the first and last columns
    for i in range(height):
        new_grid.values[2*i][0] = new_grid.values[2*i+1][0] = input_grid.values[i][0]
        new_grid.values[2*i][-1] = new_grid.values[2*i+1][-1] = input_grid.values[i][-1]
    
    # Handle the first and last rows
    for j in range(width):
        new_grid.values[0][2*j] = new_grid.values[0][2*j+1] = input_grid.values[0][j]
        new_grid.values[-1][2*j] = new_grid.values[-1][2*j+1] = input_grid.values[-1][j]
    
    # Handle the second and second-to-last columns
    for i in range(2*height):
        if i % 2 == 0:
            new_grid.values[i][1] = new_grid.values[i][0]
            new_grid.values[i][-2] = new_grid.values[i][-1]
        else:
            new_grid.values[i][1] = new_grid.values[i][2]
            new_grid.values[i][-2] = new_grid.values[i][-3]
    
    # Handle the second and second-to-last rows
    for j in range(2*width):
        if j % 2 == 0:
            new_grid.values[1][j] = new_grid.values[0][j]
            new_grid.values[-2][j] = new_grid.values[-1][j]
        else:
            new_grid.values[1][j] = new_grid.values[2][j]
            new_grid.values[-2][j] = new_grid.values[-3][j]
    
    # Handle the corners
    new_grid.values[0][0] = input_grid.values[0][0]
    new_grid.values[0][-1] = input_grid.values[0][-1]
    new_grid.values[-1][0] = input_grid.values[-1][0]
    new_grid.values[-1][-1] = input_grid.values[-1][-1]
    
    return new_grid
