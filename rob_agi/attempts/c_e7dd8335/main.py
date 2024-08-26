from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7dd8335(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing the bottom half of each continuous blue (1) region to red (2).
    The transformation is applied to each blue region separately, starting from its vertical midpoint (inclusive).
    All other colors remain unchanged.
    """
    new_grid = input_grid.deep_copy()
    blue_regions = find_blue_regions(new_grid)
    
    for region in blue_regions:
        transform_region(new_grid, region)
    
    return new_grid

def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    visited = set()
    regions = []
    rows, cols = grid.get_dimensions()
    
    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        stack = [(r, c)]
        region = []
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == 1:
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1 and (r, c) not in visited:
                regions.append(dfs(r, c))
    
    return regions

def transform_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    if not region:
        return
    
    top = min(r for r, _ in region)
    bottom = max(r for r, _ in region)
    midpoint = (top + bottom) // 2
    
    for r, c in region:
        if r > midpoint:
            grid.values[r][c] = 2  # Change to red
        elif r == midpoint:
            # If the region has an odd number of rows, change the middle row to red
            if (bottom - top + 1) % 2 != 0:
                grid.values[r][c] = 2  # Change to red
