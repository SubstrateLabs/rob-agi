from rob_agi.colored_grid import ColoredGrid
from copy import deepcopy

def solve_0ca9ddb6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific patterns around cells with values 1 and 2.
    
    The transformation rules are:
    1. For cells with value 2: Apply a diamond pattern of 4s around it.
    2. For cells with value 1: Apply a plus pattern of 7s around it.
    3. All other non-zero values remain unchanged.
    4. Patterns are only applied to cells that were originally 0.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid after applying the patterns.
    """
    def apply_pattern(grid, row, col, pattern):
        rows, cols = len(grid), len(grid[0])
        for dr, dc, value in pattern:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                grid[nr][nc] = value

    pattern_2 = [(-1, -1, 4), (-1, 1, 4), (1, -1, 4), (1, 1, 4)]
    pattern_1 = [(-1, 0, 7), (1, 0, 7), (0, -1, 7), (0, 1, 7)]
    
    output_grid = deepcopy(input_grid.values)
    
    # First pass: Apply pattern for cells with value 2
    for row in range(len(output_grid)):
        for col in range(len(output_grid[0])):
            if output_grid[row][col] == 2:
                apply_pattern(output_grid, row, col, pattern_2)
    
    # Second pass: Apply pattern for cells with value 1
    for row in range(len(output_grid)):
        for col in range(len(output_grid[0])):
            if output_grid[row][col] == 1:
                apply_pattern(output_grid, row, col, pattern_1)
    
    return ColoredGrid(values=output_grid)
