from rob_agi.colored_grid import ColoredGrid

def solve_b7f8a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colors from the centers of squares.
    Horizontal expansion occurs for yellow (4) and sky (8) colors, filling spaces within their squares.
    Vertical expansion occurs for green (3) and blue (1) colors, extending beyond their original squares.
    Expansions stop at grid edges, different colors, or when encountering vertical expansion colors for horizontal expansions.
    The grid structure (borders and frames) is preserved where possible.
    """
    height, width = len(input_grid.values), len(input_grid.values[0])
    output_grid = input_grid.deep_copy()
    
    # Identify pattern size and expansion behaviors
    pattern_size = next(i for i in range(1, width) if input_grid.values[1][i] != 0) - 1
    horizontal_colors = {4, 8}  # Yellow and Sky
    vertical_colors = {1, 3}    # Blue and Green
    
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

    def expand_vertical(row, col, color):
        for r in range(row - 1, -1, -1):
            if output_grid.values[r][col] in {0, 2, color}:  # Expand into black, red, or same color
                output_grid.values[r][col] = color
            else:
                break
        for r in range(row + 1, height):
            if output_grid.values[r][col] in {0, 2, color}:  # Expand into black, red, or same color
                output_grid.values[r][col] = color
            else:
                break
    
    # Perform expansions
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (row % (pattern_size + 1) == pattern_size // 2 + 1 and
                col % (pattern_size + 1) == pattern_size // 2 + 1):
                color = input_grid.values[row][col]
                if color in horizontal_colors:
                    expand_horizontal(row, col, color)
                elif color in vertical_colors:
                    expand_vertical(row, col, color)
    
    # Preserve grid structure
    for row in range(height):
        for col in range(width):
            if (row == 0 or row == height - 1 or col == 0 or col == width - 1 or
                row % (pattern_size + 1) == 0 or col % (pattern_size + 1) == 0):
                output_grid.values[row][col] = input_grid.values[row][col]
    
    return output_grid
