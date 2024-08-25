from rob_agi.colored_grid import ColoredGrid

def solve_506d28a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by following these steps:
    1. Extracts the top 4 rows of the input grid.
    2. Creates a new 4x5 grid, initially filled with zeros (black).
    3. For each column:
       - If there's any red cell (2) in the column, fill the entire column with green (3),
         except for the top and bottom cells.
    4. Preserves black cells (0) in the top and bottom rows if they were black in the input.
    5. Returns the transformed grid as a new ColoredGrid object.
    """
    # Extract top 4 rows
    input_top = input_grid.values[:4]
    
    # Initialize output grid
    output = [[0 for _ in range(5)] for _ in range(4)]
    
    # Process each column
    for col in range(5):
        if any(input_top[row][col] == 2 for row in range(4)):
            for row in range(1, 3):
                output[row][col] = 3
    
    # Preserve black cells in top and bottom rows
    for col in range(5):
        if input_top[0][col] != 0:
            output[0][col] = 3
        if input_top[3][col] != 0:
            output[3][col] = 3
    
    return ColoredGrid(values=output)
