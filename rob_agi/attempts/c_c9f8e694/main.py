from rob_agi.colored_grid import ColoredGrid

def solve_c9f8e694(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing all occurrences of color 5 in each row
    with the color found in the first column of that row. All other colors remain unchanged.
    
    The transformation is applied independently to each row of the grid.
    """
    rows, cols = input_grid.get_dimensions()
    output = input_grid.deep_copy()
    
    for row in range(rows):
        first_col_color = output.get_cell(row, 0)
        for col in range(cols):
            if output.get_cell(row, col) == 5:
                output.set_cell(row, col, first_col_color)
    
    return output
