from rob_agi.colored_grid import ColoredGrid

def solve_8719f442(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 15x15 output grid by:
    1. Creating a new 15x15 grid filled with 0s.
    2. Filling rows and columns with 5s based on the presence of 5s in the input.
    3. Copying the input 3x3 grid to the top-left corner.
    4. Creating a diagonal pattern if the input has 5s on the main diagonal.
    5. Creating a cross pattern if the input has a cross of 5s.
    6. Processing the bottom-right quadrant based on the input.
    7. Ensuring correct values at intersections of filled rows and columns.
    """
    # Create a new 15x15 grid filled with 0s
    new_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Fill rows and columns with 5s based on the input
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] == 5:
                for k in range(15):
                    new_grid[i*5][k] = 5
                    new_grid[k][j*5] = 5
    
    # Copy the input to the top-left corner
    for i in range(3):
        for j in range(3):
            new_grid[i][j] = input_grid.values[i][j]
    
    # Check for diagonal pattern
    if input_grid.values[0][0] == input_grid.values[1][1] == input_grid.values[2][2] == 5:
        for i in range(15):
            new_grid[i][i] = 5
            new_grid[i][i+1 if i < 14 else 14] = 5
            new_grid[i+1 if i < 14 else 14][i] = 5
    
    # Check for cross pattern
    elif input_grid.values[1][0] == input_grid.values[1][1] == input_grid.values[1][2] == input_grid.values[0][1] == input_grid.values[2][1] == 5:
        for i in range(6, 9):
            for j in range(15):
                new_grid[i][j] = 5
                new_grid[j][i] = 5
    
    # Process the bottom-right quadrant
    if input_grid.values[2][2] == 5:
        for i in range(3):
            for j in range(3):
                new_grid[10+i][10+j] = input_grid.values[i][j]
    
    # Ensure correct values at intersections
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] == 5:
                new_grid[i*5][j*5] = 5
    
    return ColoredGrid(values=new_grid)
