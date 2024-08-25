from rob_agi.colored_grid import ColoredGrid

def solve_b7f8a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colors from the centers of squares.
    Vertical expansion occurs in specific columns, while horizontal expansion
    occurs in other columns. Expansions stop at grid edges, different colors,
    or when encountering vertical expansion colors for horizontal expansions.
    The grid structure (borders and frames) is preserved where possible.
    """
    height, width = len(input_grid.values), len(input_grid.values[0])
    output_grid = input_grid.deep_copy()
    
    # Identify pattern and expansion columns
    pattern_size = next(i for i in range(1, width) if input_grid.values[1][i] != 0) - 1
    expansion_columns = set()
    for col in range(1, width - 1):
        if any(input_grid.values[row][col] in {4, 8} for row in range(height)):
            expansion_columns.add(col)
    
    def expand(row, col, color, is_vertical):
        if is_vertical:
            for r in range(row - 1, -1, -1):
                if output_grid.values[r][col] not in {0, 3, color}:
                    break
                output_grid.values[r][col] = color
            for r in range(row + 1, height):
                if output_grid.values[r][col] not in {0, 3, color}:
                    break
                output_grid.values[r][col] = color
        else:
            for c in range(col - 1, -1, -1):
                if c in expansion_columns or output_grid.values[row][c] not in {0, 3, color}:
                    break
                output_grid.values[row][c] = color
            for c in range(col + 1, width):
                if c in expansion_columns or output_grid.values[row][c] not in {0, 3, color}:
                    break
                output_grid.values[row][c] = color
    
    # Perform expansions
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (row % (pattern_size + 1) == pattern_size // 2 + 1 and
                col % (pattern_size + 1) == pattern_size // 2 + 1):
                color = input_grid.values[row][col]
                if color != 0:
                    expand(row, col, color, col in expansion_columns)
    
    # Preserve grid structure
    for row in range(height):
        for col in range(width):
            if (row == 0 or row == height - 1 or col == 0 or col == width - 1 or
                row % (pattern_size + 1) == 0 or col % (pattern_size + 1) == 0):
                output_grid.values[row][col] = input_grid.values[row][col]
    
    return output_grid
