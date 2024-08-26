from rob_agi.colored_grid import ColoredGrid

def solve_833dafe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and creating a symmetrical pattern.
    
    The function expands the input grid by doubling each cell, then mirrors
    the expanded grid both horizontally and vertically to create a symmetrical
    output that is exactly twice the size of the input in both dimensions.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*cols)] for _ in range(2*rows)])
    
    # Expand the input grid
    for i in range(rows):
        for j in range(cols):
            value = input_grid.values[i][j]
            new_grid.values[2*i][2*j] = value
            new_grid.values[2*i][2*j+1] = value
            new_grid.values[2*i+1][2*j] = value
            new_grid.values[2*i+1][2*j+1] = value
    
    # Mirror the expanded grid
    for i in range(rows):
        for j in range(cols):
            top_left = new_grid.values[i][j]
            new_grid.values[i][2*cols-1-j] = top_left  # Mirror horizontally
            new_grid.values[2*rows-1-i][j] = top_left  # Mirror vertically
            new_grid.values[2*rows-1-i][2*cols-1-j] = top_left  # Mirror diagonally
    
    return new_grid
