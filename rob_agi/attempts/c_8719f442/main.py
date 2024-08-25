from rob_agi.colored_grid import ColoredGrid

def solve_8719f442(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 15x15 output grid by:
    1. Creating a new 15x15 grid filled with 0s.
    2. Copying the input 3x3 grid to each corner of the new grid.
    3. Processing the center 7x7 area based on specific patterns in the input.
    4. Filling rows and columns with 5s based on the presence of 5s in the input.
    5. Special handling for the bottom row of the input grid.
    6. Creating diagonal patterns when top-left and bottom-right corners are 5.
    """
    # Create a new 15x15 grid filled with 0s
    new_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Copy the input 3x3 grid to each corner
    corners = [(0, 0), (0, 12), (12, 0), (12, 12)]
    for corner_row, corner_col in corners:
        for i in range(3):
            for j in range(3):
                new_grid[corner_row + i][corner_col + j] = input_grid.values[i][j]
    
    # Process the center 7x7 area
    if input_grid.values[1][1] == 5:  # Cross shape
        for i in range(4, 11):
            new_grid[i][7] = 5
            new_grid[7][i] = 5
    elif input_grid.values[0][0] == 5 and input_grid.values[2][2] == 5:  # Diagonal pattern
        for i in range(4, 11):
            new_grid[i][i] = 5
            new_grid[i][14-i] = 5
    else:  # Copy the 3x3 input pattern to the center
        for i in range(3):
            for j in range(3):
                new_grid[i + 6][j + 6] = input_grid.values[i][j]
    
    # Process rows and columns
    for i in range(3):
        if 5 in input_grid.values[i]:
            for j in range(15):
                new_grid[i * 5][j] = 5
        if 5 in [row[i] for row in input_grid.values]:
            for j in range(15):
                new_grid[j][i * 5] = 5
    
    # Special handling for the bottom row of the input
    if 5 in input_grid.values[2]:
        for j in range(15):
            new_grid[10][j] = 5
        for j in range(3):
            new_grid[10][j * 5] = input_grid.values[2][j]
    
    return ColoredGrid(values=new_grid)
