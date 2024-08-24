from rob_agi.colored_grid import ColoredGrid

def solve_22eb0ac0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling rows 3 and 7 (0-indexed) with their first non-zero number
    if the first and last numbers in the row match and are non-zero.
    All other rows remain unchanged.
    """
    output = input_grid.deep_copy()
    rows_to_check = [3, 7]
    
    for row in rows_to_check:
        first = output.get_cell(row, 0)
        last = output.get_cell(row, output.num_cols - 1)
        
        if first == last and first != 0:
            for col in range(output.num_cols):
                output.set_cell(row, col, first)
    
    return output
