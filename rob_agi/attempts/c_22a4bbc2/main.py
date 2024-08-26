from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying rectangles to red.
    
    A qualifying rectangle is a contiguous area of blue (1) or sky blue (8),
    with dimensions 2x2, 3x1, 1x3, 3x2, 2x3, 3x3, 4x1, or 1x4.
    The function identifies all such rectangles and changes them to red (color 2).
    Overlapping or adjacent qualifying rectangles are merged into larger red areas.
    All changes are applied simultaneously to the input grid.
    Non-qualifying blue or sky blue areas remain unchanged.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    marked = [[False for _ in range(cols)] for _ in range(rows)]
    qualifying_dimensions = [(2,2), (3,1), (1,3), (3,2), (2,3), (3,3), (4,1), (1,4)]
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] in [1, 8]:  # Only consider blue and sky blue
                for height, width in qualifying_dimensions:
                    if is_qualifying_shape(input_grid, r, c, height, width):
                        for i in range(height):
                            for j in range(width):
                                marked[r+i][c+j] = True
    
    for r in range(rows):
        for c in range(cols):
            if marked[r][c]:
                new_grid.values[r][c] = 2
    
    return new_grid

def is_qualifying_shape(grid: ColoredGrid, row: int, col: int, height: int, width: int) -> bool:
    rows, cols = grid.get_dimensions()
    if row + height > rows or col + width > cols:
        return False
    color = grid.values[row][col]
    if color not in [1, 8]:  # Only blue and sky blue are qualifying colors
        return False
    return all(grid.values[r][c] == color 
               for r in range(row, row + height) 
               for c in range(col, col + width))
