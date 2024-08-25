from rob_agi.colored_grid import ColoredGrid

def solve_22eb0ac0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling rows with their first non-zero number
    if the first and last non-zero numbers in the row match.
    Rows without matching non-zero numbers at the ends remain unchanged.
    """
    output = input_grid.deep_copy()
    
    for row in range(output.num_rows):
        first_non_zero = None
        last_non_zero = None
        
        for col in range(output.num_cols):
            cell = output.get_cell(row, col)
            if cell != 0:
                if first_non_zero is None:
                    first_non_zero = cell
                last_non_zero = cell
        
        if first_non_zero is not None and first_non_zero == last_non_zero:
            for col in range(output.num_cols):
                output.set_cell(row, col, first_non_zero)
    
    return output
