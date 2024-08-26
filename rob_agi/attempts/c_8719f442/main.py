from rob_agi.colored_grid import ColoredGrid

def solve_8719f442(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 15x15 output grid by:
    1. Creating a new 15x15 grid filled with 0s (black).
    2. For each gray (5) cell in the input:
       a. Fill a 5x5 section in the output grid with 5s.
       b. Extend lines of 5s from this section to all edges of the output grid.
    3. Copy the original 3x3 input pattern to the top-left corner of the output grid.
    4. If the bottom-right cell of the input is 5, copy the input pattern to the bottom-right 5x5 section.
    5. Return the completed output grid.
    """
    # Create a new 15x15 grid filled with 0s
    new_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Process each cell in the input grid
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] == 5:
                # Fill 5x5 section
                for r in range(5):
                    for c in range(5):
                        new_grid[i*5 + r][j*5 + c] = 5
                # Extend lines to edges
                for k in range(15):
                    new_grid[i*5][k] = 5
                    new_grid[k][j*5] = 5
    
    # Copy input pattern to top-left corner
    for i in range(3):
        for j in range(3):
            new_grid[i][j] = input_grid.values[i][j]
    
    # Check bottom-right cell and copy pattern if it's 5
    if input_grid.values[2][2] == 5:
        for i in range(3):
            for j in range(3):
                new_grid[10+i][10+j] = input_grid.values[i][j]
    
    return ColoredGrid(values=new_grid)
