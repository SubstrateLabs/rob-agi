from rob_agi.colored_grid import ColoredGrid

def solve_4258a5f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by surrounding each gray (5) cell with blue (1) cells.
    
    For each gray (5) cell, change all adjacent cells (including diagonals) to blue (1),
    unless they are already gray (5). This creates a 3x3 blue square with a gray center
    for each gray cell in the input. The transformation respects grid boundaries.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    def transform_surrounding(grid, row, col):
        for i in range(max(0, row-1), min(len(grid), row+2)):
            for j in range(max(0, col-1), min(len(grid[0]), col+2)):
                if grid[i][j] != 5:
                    grid[i][j] = 1
    
    grid = input_grid.deep_copy()
    for i in range(len(grid.values)):
        for j in range(len(grid.values[0])):
            if grid.values[i][j] == 5:
                transform_surrounding(grid.values, i, j)
    
    return grid
