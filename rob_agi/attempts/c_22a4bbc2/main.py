from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying rectangles to red.
    
    A qualifying rectangle is a contiguous area of blue (1) or sky blue (8),
    with dimensions 2x2, 3x1, 1x3, 3x2, 2x3, 3x3, 4x1, or 1x4.
    The function identifies all such rectangles and changes them to red (color 2).
    Overlapping or adjacent qualifying rectangles are merged into larger red areas.
    All changes are applied simultaneously to the input grid.
    Non-qualifying blue or sky blue areas remain unchanged.
    Sky blue areas are treated independently and are not changed to red unless they form a qualifying rectangle.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    qualifying_dimensions = [(2,2), (3,1), (1,3), (3,2), (2,3), (3,3), (4,1), (1,4)]
    
    to_change = set()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] in [1, 8]:  # Consider blue and sky blue
                for height, width in qualifying_dimensions:
                    if is_qualifying_shape(input_grid, r, c, height, width):
                        mark_for_change(to_change, r, c, height, width)
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] in [1, 8] and (r, c) not in to_change:
                if is_part_of_larger_rectangle(input_grid, r, c, input_grid.values[r][c]):
                    to_change.add((r, c))
    
    for r, c in to_change:
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

def mark_for_change(to_change: Set[Tuple[int, int]], row: int, col: int, height: int, width: int):
    for i in range(height):
        for j in range(width):
            to_change.add((row+i, col+j))

def is_part_of_larger_rectangle(grid: ColoredGrid, row: int, col: int, color: int) -> bool:
    rows, cols = grid.get_dimensions()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == color:
            if is_qualifying_shape(grid, min(row, nr), min(col, nc), abs(dr) + 1, abs(dc) + 1):
                return True
    
    return False
