from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors upwards within each column.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. For each column, scan from bottom to top.
    3. Keep track of the current non-black color.
    4. Fill all cells above with the current color until a new non-black color is encountered.
    5. Update the current color when a new non-black color is found.
    
    This function preserves original color positions, only extends colors upwards,
    and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    for col in range(cols):
        current_color = 0
        for row in range(rows - 1, -1, -1):
            if result.values[row][col] != 0:
                current_color = result.values[row][col]
            if current_color != 0:
                result.values[row][col] = current_color
    
    return result
