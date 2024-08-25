from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set, Dict

def solve_7c8af763(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling regions bounded by gray lines with colors.
    
    The algorithm works as follows:
    1. Identifies regions bounded by gray (5) lines.
    2. For each region:
       a. Finds the closest non-gray, non-zero color to the top-left corner of the region.
       b. Fills the region with this color, preserving original non-zero values.
    3. Preserves all gray lines and original color markers.
    
    Returns a new ColoredGrid with the transformed values.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    color_markers = find_color_markers(input_grid)
    
    for region in regions:
        fill_color = find_closest_color(region, color_markers)
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

def find_color_markers(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    rows, cols = grid.get_dimensions()
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in [0, 5]:
                markers.append((grid.values[r][c], r, c))
    return markers

def find_closest_color(region: Set[Tuple[int, int]], color_markers: List[Tuple[int, int, int]]) -> int:
    top_left = min(region)
    min_distance = float('inf')
    closest_color = None
    
    for color, r, c in color_markers:
        distance = manhattan_distance(top_left, (r, c))
        if distance < min_distance or (distance == min_distance and color < closest_color):
            min_distance = distance
            closest_color = color
    
    return closest_color

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def fill_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color: int) -> None:
    for r, c in region:
        if grid.values[r][c] == 0:
            grid.values[r][c] = color
