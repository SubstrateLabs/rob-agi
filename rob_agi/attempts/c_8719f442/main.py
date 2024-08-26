from rob_agi.colored_grid import ColoredGrid

def solve_8719f442(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 15x15 output grid by:
    1. Creating a new 15x15 grid filled with 0s (black).
    2. For each gray (5) cell in the input:
       a. Create a cross pattern in the corresponding 5x5 area of the output grid.
       b. The cross consists of a 3x3 square with additional cells at the top, bottom, left, and right.
    3. Handle overlapping areas by allowing crosses to merge into larger gray areas.
    4. Return the completed output grid.
    """
    # Create a new 15x15 grid filled with 0s (black)
    new_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Process each cell in the input grid
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] == 5:
                # Calculate the top-left corner of the corresponding 5x5 area
                output_row = i * 5
                output_col = j * 5
                
                # Create the cross pattern
                # Fill the central 3x3 square
                for r in range(3):
                    for c in range(3):
                        new_grid[output_row + 1 + r][output_col + 1 + c] = 5
                
                # Add the four extending cells
                new_grid[output_row][output_col + 2] = 5     # Top
                new_grid[output_row + 2][output_col] = 5     # Left
                new_grid[output_row + 2][output_col + 4] = 5 # Right
                new_grid[output_row + 4][output_col + 2] = 5 # Bottom
    
    return ColoredGrid(values=new_grid)
