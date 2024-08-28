from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors upwards within each column.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. For each column, scan from bottom to top.
    3. Store positions and colors of non-black cells.
    4. Process each non-black color from bottom to top:
       - Extend the color upwards through black cells.
       - Stop when reaching another non-black color or the top of the grid.
    5. Preserve original non-black color positions.
    
    This function maintains the original layout of colors, extends them upwards only through
    black cells, and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    for col in range(cols):
        color_positions = []
        for row in range(rows - 1, -1, -1):
            if result.values[row][col] != 0:
                color_positions.append((row, result.values[row][col]))
        
        for i, (start_row, color) in enumerate(color_positions):
            end_row = color_positions[i-1][0] if i > 0 else -1
            for row in range(start_row - 1, end_row, -1):
                if result.values[row][col] == 0:
                    result.values[row][col] = color
                else:
                    break
    
    return result
