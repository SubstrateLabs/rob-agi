from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x9 grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half (first 4 columns) of the input grid.
    2. Identify connection points between left and right halves.
    3. If there are connection points, expand shapes from the right half into empty spaces of the left half.
    4. If no connection points, use only the left half.
    5. Preserve the structure of the shapes, including empty spaces.
    6. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    left_half = extract_half(input_grid, 0, 4)
    right_half = extract_half(input_grid, 5, 9)
    
    result_grid = [row[:] for row in left_half]
    connection_points = find_connection_points(left_half, right_half)
    
    if connection_points:
        for r, c, color in connection_points:
            flood_fill(result_grid, right_half, r, c, color)
    
    return ColoredGrid(values=result_grid)

def extract_half(input_grid: ColoredGrid, start: int, end: int) -> List[List[int]]:
    return [row[start:end] for row in input_grid.values]

def find_connection_points(left_half: List[List[int]], right_half: List[List[int]]) -> List[Tuple[int, int, int]]:
    connection_points = []
    for r in range(4):
        if left_half[r][3] != 0 and right_half[r][0] != 0:
            connection_points.append((r, 0, right_half[r][0]))
    return connection_points

def flood_fill(result_grid: List[List[int]], right_half: List[List[int]], r: int, c: int, color: int) -> None:
    queue = deque([(r, c)])
    visited = set()

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if 0 <= r < 4 and 0 <= c < 4:
            if c < 4 and result_grid[r][c] == 0:
                result_grid[r][c] = color
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((r + dr, c + dc))
            elif c >= 4 and right_half[r][c - 4] == color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((r + dr, c + dc))
