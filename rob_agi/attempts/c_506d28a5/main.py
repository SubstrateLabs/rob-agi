from rob_agi.colored_grid import ColoredGrid

def solve_506d28a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by following these steps:
    1. Extracts the top 4 rows of the input grid.
    2. Creates a new 4x5 grid.
    3. Processes the top row: non-black cells become green, black cells remain black.
    4. Processes the bottom row: non-black cells become green, black cells remain black.
    5. For the middle two rows:
       - If there's any red cell in the column, both cells become green.
       - Otherwise, preserves the original colors from the input.
    6. Returns the transformed grid as a new ColoredGrid object.
    """
    # Extract top 4 rows
    input_top = input_grid.values[:4]
    
    # Initialize output grid
    output = [[0 for _ in range(5)] for _ in range(4)]
    
    # Process top and bottom rows
    for col in range(5):
        output[0][col] = 3 if input_top[0][col] != 0 else 0
        output[3][col] = 3 if input_top[3][col] != 0 else 0
    
    # Process middle rows
    for col in range(5):
        if any(input_top[row][col] == 2 for row in range(4)):
            output[1][col] = 3
            output[2][col] = 3
        else:
            output[1][col] = input_top[1][col]
            output[2][col] = input_top[2][col]
    
    return ColoredGrid(values=output)
