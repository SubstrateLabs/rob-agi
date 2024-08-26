from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple
from collections import Counter

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x9 grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half (first 4 columns) of the input grid.
    2. If there's a clear connection to the right half, expand the shape.
    3. If no clear connection, use only the left half.
    4. Preserve the structure of the shape, including empty spaces.
    5. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    left_half = extract_left_half(input_grid)
    right_half = extract_right_half(input_grid)
    
    if has_clear_connection(left_half, right_half):
        result_grid = expand_shape(left_half, right_half)
    else:
        result_grid = left_half
    
    return ColoredGrid(values=result_grid)

def extract_left_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[:4] for row in input_grid.values]

def extract_right_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[5:] for row in input_grid.values]

def has_clear_connection(left_half: List[List[int]], right_half: List[List[int]]) -> bool:
    left_edge = [row[3] for row in left_half]
    right_edge = [row[0] for row in right_half]
    return any(l != 0 and r != 0 for l, r in zip(left_edge, right_edge))

def expand_shape(left_half: List[List[int]], right_half: List[List[int]]) -> List[List[int]]:
    result = [row[:] for row in left_half]
    for r in range(4):
        for c in range(4):
            if right_half[r][c] != 0 and is_connected(r, c, left_half, right_half):
                result[r][c] = right_half[r][c]
    return result

def is_connected(r: int, c: int, left_half: List[List[int]], right_half: List[List[int]]) -> bool:
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            if nc < 4 and left_half[nr][nc] != 0:
                return True
            if nc >= 0 and right_half[nr][nc] != 0:
                return True
    return False
