from rob_agi.colored_grid import ColoredGrid

def solve_b7f8a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colors from the centers of squares.
    Yellow (4) and Sky (8) expand horizontally within their squares.
    Blue (1) expands vertically beyond its square until reaching a border or edge.
    Green (3) expands vertically only to the cells directly above and below.
    The grid structure (borders and frames) is preserved.
    """
    height, width = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Identify pattern size
    pattern_size = next(i for i in range(1, width) if input_grid.values[1][i] != 0) - 1
    
    def expand_horizontal(row, col, color):
        for c in range(col - 1, max(0, col - pattern_size // 2), -1):
            if output_grid.values[row][c] in {0, 2}:  # Expand into black or red
                output_grid.values[row][c] = color
            else:
                break
        for c in range(col + 1, min(width, col + pattern_size // 2 + 1)):
            if output_grid.values[row][c] in {0, 2}:  # Expand into black or red
                output_grid.values[row][c] = color
            else:
                break

    def expand_blue_vertical(row, col):
        for r in range(row - 1, -1, -1):
            if output_grid.values[r][col] in {0, 2}:  # Expand into black or red
                output_grid.values[r][col] = 1
            else:
                break
        for r in range(row + 1, height):
            if output_grid.values[r][col] in {0, 2}:  # Expand into black or red
                output_grid.values[r][col] = 1
            else:
                break

    def expand_green_limited(row, col):
        if row > 0 and output_grid.values[row-1][col] == 0:
            output_grid.values[row-1][col] = 3
        if row < height - 1 and output_grid.values[row+1][col] == 0:
            output_grid.values[row+1][col] = 3
    
    # Perform horizontal expansions
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (row % (pattern_size + 1) == pattern_size // 2 + 1 and
                col % (pattern_size + 1) == pattern_size // 2 + 1):
                color = input_grid.values[row][col]
                if color in {4, 8}:  # Yellow and Sky
                    expand_horizontal(row, col, color)
    
    # Perform vertical expansions
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (row % (pattern_size + 1) == pattern_size // 2 + 1 and
                col % (pattern_size + 1) == pattern_size // 2 + 1):
                color = input_grid.values[row][col]
                if color == 1:  # Blue
                    expand_blue_vertical(row, col)
                elif color == 3:  # Green
                    expand_green_limited(row, col)
    
    # Preserve grid structure
    for row in range(height):
        for col in range(width):
            if (row == 0 or row == height - 1 or col == 0 or col == width - 1 or
                row % (pattern_size + 1) == 0 or col % (pattern_size + 1) == 0):
                output_grid.values[row][col] = input_grid.values[row][col]
    
    return output_grid
