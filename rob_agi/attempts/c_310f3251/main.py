from rob_agi.colored_grid import ColoredGrid

def solve_310f3251(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding it 3x3 times
    2. Copying the original pattern to each expanded section
    3. Adding red squares (2) in specific positions, replacing only black squares (0)
    
    The red squares are placed in every third column of each expanded section, starting from
    a column determined by the input grid's width. They appear in a specific row within each
    section, determined by the input grid's height. Red squares are only placed if the
    corresponding position in the original pattern is black.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    input_copy = input_grid.deep_copy()
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_cols * 3)] for _ in range(input_rows * 3)])
    
    # Copy the input grid pattern to each 3x3 section of the output grid
    for i in range(input_rows * 3):
        for j in range(input_cols * 3):
            output_grid.values[i][j] = input_grid.values[i // 3][j // 3]
    
    # Determine the starting column and row for red square placement
    start_column = input_cols // 2 if input_cols % 2 == 0 else (input_cols - 1) // 2
    red_row = (input_rows // 2) - 1 if input_rows % 2 == 0 else input_rows // 2
    
    # Add red squares to the appropriate positions
    for section_row in range(3):
        for section_col in range(3):
            for col in range(start_column, input_cols, 3):
                output_row = section_row * input_rows + red_row
                output_col = section_col * input_cols + col
                if (output_grid.values[output_row][output_col] == 0 and
                    input_copy.values[red_row][col] == 0):
                    output_grid.values[output_row][output_col] = 2  # 2 represents red
    
    return output_grid
