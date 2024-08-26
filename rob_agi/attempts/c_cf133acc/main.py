from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors upwards within each column.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. For each column, scan from bottom to top.
    3. When a non-black color is found, extend it upwards, filling black cells.
    4. Stop extension when encountering another non-black color or the top of the grid.
    
    This function preserves original color positions, only extends colors upwards,
    and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    for col in range(cols):
        for row in range(rows - 1, -1, -1):
            color = result.values[row][col]
            
            if color != 0:
                for up_row in range(row - 1, -1, -1):
                    if result.values[up_row][col] != 0:
                        break
                    result.values[up_row][col] = color
    
    return result
