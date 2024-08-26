from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by replicating hollow square patterns within framed areas.
    
    The function identifies framed areas in the input grid, finds hollow squares within each area,
    replicates them horizontally and vertically within the bounds of the frame, preserves the
    original frame lines, and keeps unframed areas unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with replicated square patterns within framed areas.
    """
    framed_areas = find_framed_areas(input_grid)
    new_grid = [row[:] for row in input_grid.values]
    
    for top, bottom, left, right in framed_areas:
        squares = find_hollow_squares(input_grid, top, bottom, left, right)
        if squares:
            replicate_squares(new_grid, squares, top, bottom, left, right)
    
    preserve_frame_lines(new_grid, input_grid)
    return ColoredGrid(values=new_grid)

def find_framed_areas(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    framed_areas = []
    rows, cols = len(grid.values), len(grid.values[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if grid.values[i][j] != 0 and not visited[i][j]:
                top, bottom, left, right = i, i, j, j
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] != 0 and not visited[r][c]:
                        visited[r][c] = True
                        top, bottom = min(top, r), max(bottom, r)
                        left, right = min(left, c), max(right, c)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                framed_areas.append((top, bottom, left, right))
    
    return framed_areas

def find_hollow_squares(grid: ColoredGrid, top: int, bottom: int, left: int, right: int) -> List[Tuple[int, int, int]]:
    squares = []
    for i in range(top, bottom - 2):
        for j in range(left, right - 2):
            if is_hollow_square(grid, i, j):
                squares.append((i, j, grid.values[i][j]))
    return squares

def is_hollow_square(grid: ColoredGrid, i: int, j: int) -> bool:
    color = grid.values[i][j]
    return (all(grid.values[i+di][j+dj] == color for di, dj in [(0,0), (0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2), (3,3)]) and
            all(grid.values[i+di][j+dj] == 0 for di, dj in [(1,1), (1,2), (2,1), (2,2)]))

def replicate_squares(new_grid: List[List[int]], squares: List[Tuple[int, int, int]], top: int, bottom: int, left: int, right: int):
    for row in range(top, bottom + 1, 4):
        for col in range(left, right + 1, 4):
            for i, j, color in squares:
                for di, dj in [(0,0), (0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2), (3,3)]:
                    if top <= row + di <= bottom and left <= col + dj <= right:
                        new_grid[row + di][col + dj] = color

def preserve_frame_lines(new_grid: List[List[int]], original_grid: ColoredGrid):
    for i in range(len(original_grid.values)):
        for j in range(len(original_grid.values[0])):
            if original_grid.values[i][j] != 0:
                new_grid[i][j] = original_grid.values[i][j]
