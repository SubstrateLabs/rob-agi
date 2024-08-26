from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors vertically within each column.
    
    The function performs the following steps:
    1. For each column, identify color segments (consecutive cells of the same non-black color).
    2. For each color segment, extend the color upwards and downwards, filling black cells.
    3. Stop extension when encountering another non-black color or the edge of the grid.
    
    This function preserves original color patterns and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()

    for col in range(cols):
        segments = []
        current_color = result.values[0][col]
        start = 0
        
        # Identify color segments in the column
        for row in range(rows):
            if result.values[row][col] != current_color or row == rows - 1:
                end = row - 1 if result.values[row][col] != current_color else row
                segments.append((current_color, start, end))
                current_color = result.values[row][col]
                start = row
        
        # Extend each non-black color segment
        for color, start, end in segments:
            if color != 0:  # If not black
                # Extend upwards
                for row in range(start - 1, -1, -1):
                    if result.values[row][col] != 0:
                        break
                    result.values[row][col] = color
                
                # Extend downwards
                for row in range(end + 1, rows):
                    if result.values[row][col] != 0:
                        break
                    result.values[row][col] = color

    return result
