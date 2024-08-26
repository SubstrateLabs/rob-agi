from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying rectangles to red.
    
    A qualifying rectangle is a contiguous area of the same color (except black),
    with dimensions 2x1, 1x2, 2x2, 3x1, or 1x3.
    The function identifies all such rectangles and changes them to red (color 2).
    Overlapping or adjacent qualifying rectangles are merged into larger red areas.
    All changes are applied simultaneously to the input grid.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    marked = [[False for _ in range(cols)] for _ in range(rows)]
    qualifying_dimensions = [(2,1), (1,2), (2,2), (3,1), (1,3)]
    
    for r in range(rows):
        for c in range(cols):
            for height, width in qualifying_dimensions:
                if is_qualifying_rectangle(input_grid, r, c, height, width):
                    for i in range(height):
                        for j in range(width):
                            marked[r+i][c+j] = True
    
    for r in range(rows):
        for c in range(cols):
            if marked[r][c] and input_grid.values[r][c] != 0:
                new_grid.values[r][c] = 2
    
    return new_grid

def is_qualifying_rectangle(grid: ColoredGrid, row: int, col: int, height: int, width: int) -> bool:
    rows, cols = grid.get_dimensions()
    if row + height > rows or col + width > cols:
        return False
    color = grid.values[row][col]
    if color == 0:  # black
        return False
    return all(grid.values[r][c] == color 
               for r in range(row, row + height) 
               for c in range(col, col + width))
