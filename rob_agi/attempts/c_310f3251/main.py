from rob_agi.colored_grid import ColoredGrid

def solve_310f3251(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding it 3x3 times
    2. Copying the original pattern to each expanded section
    3. Adding red squares (2) in a diagonal pattern, replacing only black squares (0)
    
    The diagonal pattern is determined by the formula:
    (original_row + original_column) % input_rows == input_rows - 1
    """
    input_rows, input_cols = input_grid.get_dimensions()
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_cols * 3)] for _ in range(input_rows * 3)])
    
    for i in range(input_rows * 3):
        for j in range(input_cols * 3):
            output_grid.values[i][j] = input_grid.values[i % input_rows][j % input_cols]
    
    for i in range(input_rows * 3):
        for j in range(input_cols * 3):
            orig_i = i % input_rows
            orig_j = j % input_cols
            if (orig_i + orig_j) % input_rows == input_rows - 1:
                if output_grid.values[i][j] == 0:  # Only replace black squares
                    output_grid.values[i][j] = 2  # 2 represents red
    
    return output_grid
