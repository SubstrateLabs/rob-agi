from rob_agi.colored_grid import ColoredGrid

def solve_310f3251(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding it 3x3 times
    2. Copying the original pattern to each expanded section
    3. Adding red squares (2) in specific columns, replacing only black squares (0)
    
    The red squares are placed in every third column of the output grid, starting from
    the third column of each expanded input column. They appear every N rows, where N
    is the height of the input grid.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_cols * 3)] for _ in range(input_rows * 3)])
    
    # Copy the input grid pattern to each 3x3 section of the output grid
    for i in range(input_rows * 3):
        for j in range(input_cols * 3):
            output_grid.values[i][j] = input_grid.values[i // 3][j // 3]
    
    # Add red squares to the appropriate positions
    for col in range(2, input_cols * 3, 3):
        for row in range(0, input_rows * 3, input_rows):
            if output_grid.values[row][col] == 0:
                output_grid.values[row][col] = 2  # 2 represents red
    
    return output_grid
