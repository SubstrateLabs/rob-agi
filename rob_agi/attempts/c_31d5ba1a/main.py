from rob_agi.colored_grid import ColoredGrid

def solve_31d5ba1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x5 input grid into a 3x5 output grid based on the following rules:
    1. For each column in the input grid:
       - If BOTH cells in rows 0-1 contain 9, set the corresponding cell in the first row of the output to 6.
       - If EITHER cell in rows 2-3 contains 9 OR 4, set the corresponding cell in the second row of the output to 6.
       - If EITHER cell in rows 4-5 contains 4, set the corresponding cell in the third row of the output to 6.
    2. All other cells in the output grid remain 0 (black).
    """
    # Initialize the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(3)])
    
    # Process the input grid
    for col in range(5):
        # Check top section (rows 0 and 1)
        if input_grid.values[0][col] == 9 and input_grid.values[1][col] == 9:
            output_grid.values[0][col] = 6
        
        # Check middle section (rows 2 and 3)
        if input_grid.values[2][col] in [9, 4] or input_grid.values[3][col] in [9, 4]:
            output_grid.values[1][col] = 6
        
        # Check bottom section (rows 4 and 5)
        if input_grid.values[4][col] == 4 or input_grid.values[5][col] == 4:
            output_grid.values[2][col] = 6
    
    return output_grid
