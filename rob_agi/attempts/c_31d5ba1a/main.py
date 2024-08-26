from rob_agi.colored_grid import ColoredGrid

def solve_31d5ba1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x5 input grid into a 3x5 output grid based on the following rules:
    1. For each column in the input grid:
       - If EITHER cell in rows 0-1 contains 9 (brown), set the corresponding cell in the first row of the output to 6 (magenta).
       - If ANY cell in rows 0-3 contains 9 (brown), set the corresponding cell in the second row of the output to 6 (magenta).
       - If EITHER cell in rows 4-5 contains 4 (yellow), set the corresponding cell in the third row of the output to 6 (magenta).
    2. All other cells in the output grid remain 0 (black).
    3. Each column in the output is determined independently based on the corresponding column in the input.
    """
    # Initialize the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(3)])
    
    # Process each column
    for col in range(5):
        # Check first row of output (rows 0-1 of input)
        if 9 in [input_grid.values[0][col], input_grid.values[1][col]]:
            output_grid.values[0][col] = 6
        
        # Check second row of output (rows 0-3 of input)
        if 9 in [input_grid.values[0][col], input_grid.values[1][col], input_grid.values[2][col], input_grid.values[3][col]]:
            output_grid.values[1][col] = 6
        
        # Check third row of output (rows 4-5 of input)
        if 4 in [input_grid.values[4][col], input_grid.values[5][col]]:
            output_grid.values[2][col] = 6
    
    return output_grid
