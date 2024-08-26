from rob_agi.colored_grid import ColoredGrid

def solve_833dafe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and creating a symmetrical pattern.
    
    The function creates an output grid twice the size of the input in both dimensions.
    It fills the columns by duplicating each input column, handles the top and bottom rows separately,
    and ensures both vertical and horizontal symmetry. The corners are given special treatment to
    maintain the pattern. The result is a symmetrical pattern that expands on the input grid's design.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*width)] for _ in range(2*height)])
    
    # Fill the columns
    for i in range(height):
        for j in range(width):
            new_grid.values[i][2*j] = input_grid.values[i][j]
            new_grid.values[i][2*j+1] = input_grid.values[i][j]
    
    # Handle top and bottom rows
    for j in range(2*width):
        new_grid.values[0][j] = input_grid.values[0][j//2]
        new_grid.values[2*height-1][j] = input_grid.values[height-1][j//2]
    
    # Handle second and second-to-last rows
    for j in range(2*width):
        if j % 2 == 0:
            new_grid.values[1][j] = input_grid.values[1][j//2]
            new_grid.values[2*height-2][j] = input_grid.values[height-2][j//2]
        else:
            new_grid.values[1][j] = new_grid.values[0][j]
            new_grid.values[2*height-2][j] = new_grid.values[2*height-1][j]
    
    # Handle middle rows
    for i in range(2, 2*height-2):
        for j in range(2*width):
            if j == 0 or j == 2*width-1:
                new_grid.values[i][j] = input_grid.values[i//2][j//2]
            else:
                new_grid.values[i][j] = new_grid.values[1][j]
    
    # Ensure vertical symmetry
    for i in range(height):
        for j in range(2*width):
            new_grid.values[2*height-1-i][j] = new_grid.values[i][j]
    
    # Ensure horizontal symmetry
    for i in range(2*height):
        for j in range(width):
            new_grid.values[i][2*width-1-j] = new_grid.values[i][j]
    
    # Handle corners
    new_grid.values[0][0] = input_grid.values[0][0]
    new_grid.values[0][2*width-1] = input_grid.values[0][width-1]
    new_grid.values[2*height-1][0] = input_grid.values[height-1][0]
    new_grid.values[2*height-1][2*width-1] = input_grid.values[height-1][width-1]
    
    return new_grid
