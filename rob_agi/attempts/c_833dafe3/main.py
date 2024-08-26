from rob_agi.colored_grid import ColoredGrid

def solve_833dafe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and creating a symmetrical pattern.
    
    The function creates an output grid twice the size of the input in both dimensions.
    It fills the top-left quarter by duplicating each column horizontally, then mirrors
    this quarter horizontally to fill the top-right quarter. The entire top half is then
    mirrored vertically to fill the bottom half. The edges are handled separately to
    ensure correct symmetry. The result is a symmetrical pattern that preserves and
    expands the characteristics of the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*cols)] for _ in range(2*rows)])
    
    # Fill the top-left quarter
    for i in range(rows):
        for j in range(cols):
            new_grid.values[i][2*j] = input_grid.values[i][j]
            new_grid.values[i][2*j+1] = input_grid.values[i][j]
    
    # Mirror the top-left quarter horizontally
    for i in range(rows):
        for j in range(cols):
            new_grid.values[i][2*cols-1-j] = new_grid.values[i][j]
    
    # Mirror the entire top half vertically
    for i in range(rows):
        for j in range(2*cols):
            new_grid.values[2*rows-1-i][j] = new_grid.values[i][j]
    
    # Handle the edges
    for j in range(2*cols):
        new_grid.values[0][j] = input_grid.values[0][j//2]
        new_grid.values[2*rows-1][j] = input_grid.values[rows-1][j//2]
    
    for i in range(2*rows):
        new_grid.values[i][0] = input_grid.values[i//2][0]
        new_grid.values[i][2*cols-1] = input_grid.values[i//2][cols-1]
    
    return new_grid
