from rob_agi.colored_grid import ColoredGrid

def solve_d037b0a7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating non-zero values downwards in each column.
    For each column:
    1. Start from the top (first row).
    2. Keep track of the last non-zero value encountered.
    3. Replace zeros below with the last non-zero value.
    4. Update the tracker when a new non-zero value is found.
    """
    result = input_grid.deep_copy()
    height, width = result.get_dimensions()

    for col in range(width):
        last_non_zero = result.get_cell(0, col)
        
        for row in range(1, height):
            current_value = result.get_cell(row, col)
            
            if current_value == 0 and last_non_zero != 0:
                result.set_cell(row, col, last_non_zero)
            elif current_value != 0:
                last_non_zero = current_value

    return result
