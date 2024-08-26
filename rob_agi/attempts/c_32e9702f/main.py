from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_32e9702f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Replace all black (0) cells with gray (5) cells.
    2. Preserve all non-black shapes (connected regions of non-black cells).
    3. If there's a 2x2 yellow (4) square in the top-left corner, expand it diagonally by one cell.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Create a deep copy
    result_grid = input_grid.deep_copy()
    
    # Step 2: Handle yellow expansion
    handle_yellow_expansion(result_grid)
    
    # Step 3: Identify non-black shapes
    non_black_regions = identify_non_black_regions(result_grid)
    
    # Step 4: Replace black with gray, preserve non-black shapes
    replace_black_with_gray(result_grid, non_black_regions)
    
    return result_grid

def handle_yellow_expansion(grid: ColoredGrid):
    if (len(grid.values) > 2 and len(grid.values[0]) > 2 and
        grid.values[0][0] == 4 and grid.values[0][1] == 4 and
        grid.values[1][0] == 4 and grid.values[1][1] == 4):
        grid.values[2][2] = 4

def identify_non_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = len(grid.values), len(grid.values[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    regions = []

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            visited[r][c] or grid.values[r][c] != color):
            return []
        
        visited[r][c] = True
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc, color))
        return region

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid.values[r][c] != 0:
                regions.append(dfs(r, c, grid.values[r][c]))

    return regions

def replace_black_with_gray(grid: ColoredGrid, non_black_regions: List[List[Tuple[int, int]]]):
    rows, cols = len(grid.values), len(grid.values[0])
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:  # If black
                grid.values[r][c] = 5   # Change to gray
