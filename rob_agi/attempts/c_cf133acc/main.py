from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending vertical lines and then horizontal lines.
    
    1. Process vertical lines: Extend non-black colors vertically (both up and down).
    2. Process horizontal lines: Extend the rightmost non-black color horizontally to the right.
    3. Process vertical lines again to fill any remaining gaps.
    
    This function preserves original colors and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()

    def vertical_extension():
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

    # First vertical extension
    vertical_extension()

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

    # Second vertical extension to fill any remaining gaps
    vertical_extension()

    return result
