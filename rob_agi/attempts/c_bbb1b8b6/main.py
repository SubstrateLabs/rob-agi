from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x9 grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half (first 4 columns) and right half (last 4 columns) of the input grid.
    2. Identify shapes in the right half that touch the dividing line.
    3. For each identified shape, expand it into empty spaces of the left half, preserving its structure.
    4. If no expansion is possible, use only the left half.
    5. Preserve the structure of the original shapes in the left half.
    6. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    left_half = extract_half(input_grid, 0, 4)
    right_half = extract_half(input_grid, 5, 9)
    
    result_grid = [row[:] for row in left_half]
    shapes = identify_shapes(right_half)
    
    for color, shape in shapes:
        expand_shape(result_grid, shape, color)
    
    return ColoredGrid(values=result_grid)

def extract_half(input_grid: ColoredGrid, start: int, end: int) -> List[List[int]]:
    return [row[start:end] for row in input_grid.values]

def identify_shapes(right_half: List[List[int]]) -> List[Tuple[int, Set[Tuple[int, int]]]]:
    shapes = []
    visited = set()
    
    for r in range(4):
        if right_half[r][0] != 0 and (r, 0) not in visited:
            color = right_half[r][0]
            shape = set()
            queue = deque([(r, 0)])
            
            while queue:
                cr, cc = queue.popleft()
                if (cr, cc) in visited or right_half[cr][cc] != color:
                    continue
                
                visited.add((cr, cc))
                shape.add((cr, cc))
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        queue.append((nr, nc))
            
            shapes.append((color, shape))
    
    return shapes

def expand_shape(result_grid: List[List[int]], shape: Set[Tuple[int, int]], color: int) -> None:
    left_edge = min(c for _, c in shape)
    top_edge = min(r for r, _ in shape)
    bottom_edge = max(r for r, _ in shape)
    shape_height = bottom_edge - top_edge + 1
    
    for start_col in range(3, -1, -1):
        if can_expand(result_grid, shape, top_edge, start_col):
            for r, c in shape:
                new_r = r - top_edge
                new_c = start_col + (c - left_edge)
                if 0 <= new_r < 4 and 0 <= new_c < 4:
                    result_grid[new_r][new_c] = color
            break

def can_expand(result_grid: List[List[int]], shape: Set[Tuple[int, int]], top_edge: int, start_col: int) -> bool:
    left_edge = min(c for _, c in shape)
    for r, c in shape:
        new_r = r - top_edge
        new_c = start_col + (c - left_edge)
        if new_r < 0 or new_r >= 4 or new_c < 0 or new_c >= 4 or result_grid[new_r][new_c] != 0:
            return False
    return True
