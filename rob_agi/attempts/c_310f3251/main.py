from rob_agi.colored_grid import ColoredGrid

def solve_310f3251(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding it 3x3 times
    2. Copying the original pattern to each expanded section
    3. Adding red squares (2) in a specific column, replacing only black squares (0)
    
    The red squares are placed in the column (input_cols - 1) of the output grid,
    appearing every N rows, where N is the height of the input grid.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_cols * 3)] for _ in range(input_rows * 3)])
    
    for i in range(input_rows * 3):
        for j in range(input_cols * 3):
            output_grid.values[i][j] = input_grid.values[i % input_rows][j % input_cols]
    
    red_column = input_cols - 1
    for i in range(0, input_rows * 3, input_rows):
        if output_grid.values[i][red_column] == 0:
            output_grid.values[i][red_column] = 2  # 2 represents red
    
    return output_grid
