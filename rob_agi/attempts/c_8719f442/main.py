from rob_agi.colored_grid import ColoredGrid

def solve_8719f442(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 15x15 output grid by:
    1. Creating a new 15x15 grid filled with 0s (black).
    2. For each gray (5) cell in the input:
       a. Draw a 3-cell wide vertical line in the corresponding column of the output grid.
       b. Draw a 3-cell wide horizontal line in the corresponding row of the output grid.
    3. Return the completed output grid.
    """
    # Create a new 15x15 grid filled with 0s
    new_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Process each cell in the input grid
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] == 5:
                # Calculate corresponding column and row in output grid
                output_col = j * 5
                output_row = i * 5
                
                # Draw vertical line
                for row in range(15):
                    new_grid[row][output_col] = 5
                    new_grid[row][output_col + 1] = 5
                    new_grid[row][output_col + 2] = 5
                
                # Draw horizontal line
                for col in range(15):
                    new_grid[output_row][col] = 5
                    new_grid[output_row + 1][col] = 5
                    new_grid[output_row + 2][col] = 5
    
    return ColoredGrid(values=new_grid)
