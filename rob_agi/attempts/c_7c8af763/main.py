from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set

def solve_7c8af763(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling regions bounded by gray lines with colors.
    
    The algorithm works as follows:
    1. Identifies regions bounded by gray (5) lines.
    2. For each region:
       a. Finds the corners of the region.
       b. Determines the fill color based on the first non-zero, non-gray color found in the corners.
       c. Fills the region with this color, preserving original non-zero values.
    3. Preserves all gray lines and original color markers.
    
    Returns a new ColoredGrid with the transformed values.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(output_grid)
    
    for region in regions:
        corners = get_region_corners(region)
        fill_color = get_fill_color(output_grid, corners)
        fill_region(output_grid, region, fill_color)
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] == 0:
                region = bfs_region(grid, r, c)
                regions.append(region)
                visited.update(region)
    
    return regions

def bfs_region(grid: ColoredGrid, start_r: int, start_c: int) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    queue = deque([(start_r, start_c)])
    region = set()
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in region or grid.values[r][c] == 5:
            continue
        region.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] != 5:
                queue.append((nr, nc))
    
    return region

def get_region_corners(region: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]

def get_fill_color(grid: ColoredGrid, corners: List[Tuple[int, int]]) -> int:
    for r, c in corners:
        if grid.values[r][c] not in [0, 5]:
            return grid.values[r][c]
    return 0  # Default to black if no color found

def fill_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color: int) -> None:
    for r, c in region:
        if grid.values[r][c] == 0:
            grid.values[r][c] = color
