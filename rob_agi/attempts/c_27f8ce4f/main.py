from rob_agi.colored_grid import ColoredGrid

def solve_27f8ce4f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by replicating the input.
    
    The input is placed in three positions of the output grid:
    1. Center-left (rows 3-5, columns 0-2)
    2. Center (rows 3-5, columns 3-5)
    3. Bottom-center (rows 6-8, columns 3-5)
    
    The remaining spaces are filled with zeros (black/empty).
    """
    # Initialize new 9x9 grid
    new_grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # Extract input 3x3 grid values
    input_values = input_grid.values
    
    # Define placement functions
    def place_center_left():
        for i in range(3):
            for j in range(3):
                new_grid[i+3][j] = input_values[i][j]
    
    def place_center():
        for i in range(3):
            for j in range(3):
                new_grid[i+3][j+3] = input_values[i][j]
    
    def place_bottom_center():
        for i in range(3):
            for j in range(3):
                new_grid[i+6][j+3] = input_values[i][j]
    
    # Place input in all required positions
    place_center_left()
    place_center()
    place_bottom_center()
    
    # Return new ColoredGrid
    return ColoredGrid(values=new_grid)
