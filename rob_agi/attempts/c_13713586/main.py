from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions vertically and then horizontally,
    while preserving original boundaries and respecting the order of expansion.
    
    The algorithm works as follows:
    1. Create a copy of the input grid.
    2. Identify all colored positions (excluding black and gray).
    3. Sort colored positions from top to bottom, then left to right.
    4. Expand each color vertically, respecting original boundaries.
    5. Expand each color horizontally, respecting original boundaries and other colors.
    6. Repeat horizontal expansion until no changes occur or max iterations reached.
    7. Preserve all original non-black boundaries throughout the process.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def expand_vertically(grid: ColoredGrid, row: int, col: int, color: int):
        # Expand upwards
        for r in range(row-1, -1, -1):
            if input_grid.values[r][col] != 0:
                break
            grid.values[r][col] = color
        
        # Expand downwards
        for r in range(row+1, rows):
            if input_grid.values[r][col] != 0:
                break
            grid.values[r][col] = color

    def expand_horizontally(grid: ColoredGrid, row: int, col: int, color: int):
        # Expand left
        for c in range(col-1, -1, -1):
            if input_grid.values[row][c] != 0 or grid.values[row][c] not in [0, color]:
                break
            grid.values[row][c] = color
        
        # Expand right
        for c in range(col+1, cols):
            if input_grid.values[row][c] != 0 or grid.values[row][c] not in [0, color]:
                break
            grid.values[row][c] = color

    colored_positions = [(r, c, grid.values[r][c]) for r in range(rows) 
                         for c in range(cols) 
                         if grid.values[r][c] not in [0, 5]]
    colored_positions.sort()  # Sort by row, then column
    
    # Vertical expansion
    for row, col, color in colored_positions:
        expand_vertically(grid, row, col, color)
    
    # Horizontal expansion
    max_iterations = 10  # Increased to ensure full expansion
    for _ in range(max_iterations):
        changed = False
        for row, col, color in colored_positions:
            old_values = [row[:] for row in grid.values]
            expand_horizontally(grid, row, col, color)
            if grid.values != old_values:
                changed = True
        if not changed:
            break
    
    # Preserve original non-black boundaries
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                grid.values[r][c] = input_grid.values[r][c]

    return grid
