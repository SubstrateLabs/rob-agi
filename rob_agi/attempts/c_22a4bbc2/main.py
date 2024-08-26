from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying rectangles to red.
    
    A qualifying rectangle is a contiguous area of the same color (except black),
    with dimensions 2x1, 1x2, 2x2, 3x1, 1x3, 2x3, or 3x2.
    The function identifies all such rectangles and changes them to red (color 2).
    Overlapping or adjacent qualifying rectangles are merged into larger red areas.
    """
    new_grid = input_grid.deep_copy()
    cells_to_change = find_qualifying_rectangles(input_grid)
    
    for row, col in cells_to_change:
        new_grid.values[row][col] = 2
    
    return new_grid

def find_qualifying_rectangles(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    cells_to_change = set()
    qualifying_dimensions = [(2,1), (1,2), (2,2), (3,1), (1,3), (2,3), (3,2)]

    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            if color == 0:  # Skip black (empty) cells
                continue
            for height, width in qualifying_dimensions:
                if is_qualifying_rectangle(grid, r, c, height, width, color):
                    for i in range(height):
                        for j in range(width):
                            cells_to_change.add((r+i, c+j))

    return cells_to_change

def is_qualifying_rectangle(grid: ColoredGrid, row: int, col: int, height: int, width: int, color: int) -> bool:
    rows, cols = grid.get_dimensions()
    if row + height > rows or col + width > cols:
        return False
    return all(grid.values[r][c] == color 
               for r in range(row, row + height) 
               for c in range(col, col + width))
