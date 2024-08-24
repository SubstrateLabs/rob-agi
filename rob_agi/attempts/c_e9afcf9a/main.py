from rob_agi.colored_grid import ColoredGrid

def solve_e9afcf9a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by alternating values from the two input rows.
    Even columns are filled with the first element of the corresponding input row,
    while odd columns are filled with the first element of the other input row.
    This pattern is alternated for each row of the output grid.
    """
    rows, cols = input_grid.get_dimensions()
    output = input_grid.deep_copy()
    
    for i in range(rows):
        for j in range(cols):
            if j % 2 == 0:
                output.set_cell(i, j, input_grid.get_cell(i, 0))
            else:
                output.set_cell(i, j, input_grid.get_cell((i + 1) % rows, 0))
    
    return output
