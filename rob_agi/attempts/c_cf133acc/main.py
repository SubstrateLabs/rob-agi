from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending vertical lines and then horizontal lines.
    
    1. Perform vertical extension: Extend non-black colors vertically (both up and down),
       stopping at other colors.
    2. Perform horizontal extension: Extend the rightmost non-black color horizontally
       to the right, filling only black cells.
    
    This function preserves original colors and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()

    # Vertical extension
    for c in range(cols):
        # Top-to-bottom pass
        last_color = 0
        for r in range(rows):
            if result.values[r][c] != 0:
                last_color = result.values[r][c]
            elif last_color != 0:
                result.values[r][c] = last_color

        # Bottom-to-top pass
        last_color = 0
        for r in range(rows-1, -1, -1):
            if result.values[r][c] != 0:
                last_color = result.values[r][c]
            elif last_color != 0:
                result.values[r][c] = last_color

    # Horizontal extension
    for r in range(rows):
        rightmost_color = 0
        for c in range(cols-1, -1, -1):
            if result.values[r][c] != 0:
                rightmost_color = result.values[r][c]
                break
        
        if rightmost_color != 0:
            for c in range(cols):
                if result.values[r][c] == 0:
                    result.values[r][c] = rightmost_color

    return result
