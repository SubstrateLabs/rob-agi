from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending horizontal and vertical lines.
    
    1. Process horizontal lines: Extend non-black colors vertically.
    2. Process vertical lines: Extend non-black colors horizontally to the right.
    
    This function preserves original colors and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()

    # Process horizontal lines
    for r in range(rows):
        for c in range(cols):
            if result.values[r][c] != 0:
                color = result.values[r][c]
                # Extend upwards
                for up in range(r-1, -1, -1):
                    if result.values[up][c] != 0:
                        break
                    result.values[up][c] = color
                # Extend downwards
                for down in range(r+1, rows):
                    if result.values[down][c] != 0:
                        break
                    result.values[down][c] = color

    # Process vertical lines
    for c in range(cols):
        for r in range(rows):
            if result.values[r][c] != 0:
                color = result.values[r][c]
                # Extend to the right
                for right in range(c+1, cols):
                    if result.values[r][right] != 0:
                        break
                    result.values[r][right] = color

    return result
