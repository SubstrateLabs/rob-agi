from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending colors vertically and then horizontally.
    
    1. Perform vertical extension: Extend non-black colors vertically (both up and down),
       filling black cells with the nearest non-black color in the column.
    2. Perform horizontal extension: Extend non-black colors horizontally to the right,
       filling black cells with the nearest non-black color to the left in the row.
    
    This function preserves original color patterns and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()

    # Vertical Extension
    for c in range(cols):
        # Top to bottom
        current_color = 0
        for r in range(rows):
            if result.values[r][c] != 0:
                current_color = result.values[r][c]
            elif current_color != 0:
                result.values[r][c] = current_color
        
        # Bottom to top
        current_color = 0
        for r in range(rows-1, -1, -1):
            if result.values[r][c] != 0:
                current_color = result.values[r][c]
            elif current_color != 0:
                result.values[r][c] = current_color

    # Horizontal Extension
    for r in range(rows):
        current_color = 0
        for c in range(cols):
            if result.values[r][c] != 0:
                current_color = result.values[r][c]
            elif current_color != 0:
                result.values[r][c] = current_color

    return result
