from rob_agi.colored_grid import ColoredGrid

def solve_31d5ba1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x5 input grid into a 3x5 output grid based on the following rules:
    1. The input grid is divided into three sections of two rows each.
    2. For each column in the input grid:
       - If EITHER cell in the top section (rows 0-1) contains 9 (brown), set the corresponding cell in the first row of the output to 6 (magenta).
       - If EITHER cell in the middle section (rows 2-3) contains 9 (brown), set the corresponding cell in the second row of the output to 6 (magenta).
       - If EITHER cell in the bottom section (rows 4-5) contains 4 (yellow), set the corresponding cell in the third row of the output to 6 (magenta).
    3. All other cells in the output grid remain 0 (black).
    4. Each column in the output is determined independently based on the corresponding column in the input.
    5. The transformation is applied consistently across all columns.
    """
    # Initialize the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(3)])
    
    # Process each column
    for col in range(5):
        # Check top section (rows 0 and 1)
        if input_grid.values[0][col] == 9 or input_grid.values[1][col] == 9:
            output_grid.values[0][col] = 6
        
        # Check middle section (rows 2 and 3)
        if input_grid.values[2][col] == 9 or input_grid.values[3][col] == 9:
            output_grid.values[1][col] = 6
        
        # Check bottom section (rows 4 and 5)
        if input_grid.values[4][col] == 4 or input_grid.values[5][col] == 4:
            output_grid.values[2][col] = 6
    
    return output_grid
