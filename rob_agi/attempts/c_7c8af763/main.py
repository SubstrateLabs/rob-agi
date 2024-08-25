from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set, Dict

def solve_7c8af763(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling regions based on color influence from the edges.
    
    The algorithm works as follows:
    1. Creates an influence map based on colors propagating from the edges.
    2. Identifies regions bounded by gray (5) lines or non-zero colors.
    3. Fills each region with the color from the influence map.
    4. Preserves all gray lines and original non-zero color markers.
    
    Returns a new ColoredGrid with the transformed values.
    """
    influence_map = create_influence_map(input_grid)
    output_grid = input_grid.deep_copy()
    regions = find_regions(output_grid)
    
    for region in regions:
        fill_color = get_influence_color(influence_map, region)
        fill_region(output_grid, region, fill_color)
    
    return output_grid

def create_influence_map(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = grid.get_dimensions()
    influence_map = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def propagate(r: int, c: int, color: int):
        queue = deque([(r, c)])
        visited = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or grid.values[r][c] not in [0, 5]:
                continue
            visited.add((r, c))
            influence_map[r][c] = color
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))
    
    # Propagate from edges
    for r in [0, rows-1]:
        for c in range(cols):
            if grid.values[r][c] not in [0, 5]:
                propagate(r, c, grid.values[r][c])
    
    for c in [0, cols-1]:
        for r in range(rows):
            if grid.values[r][c] not in [0, 5]:
                propagate(r, c, grid.values[r][c])
    
    return influence_map

def find_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] in [0, 5]:
                region = bfs_region(grid, r, c)
                regions.append(region)
                visited.update(region)
    
    return regions

def bfs_region(grid: ColoredGrid, start_r: int, start_c: int) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    queue = deque([(start_r, start_c)])
    region = set()
    start_value = grid.values[start_r][start_c]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in region or grid.values[r][c] not in [0, 5]:
            continue
        region.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in [0, 5]:
                queue.append((nr, nc))
    
    return region

def get_influence_color(influence_map: List[List[int]], region: Set[Tuple[int, int]]) -> int:
    for r, c in region:
        if influence_map[r][c] != 0:
            return influence_map[r][c]
    return 0  # Default to black if no influence found

def fill_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color: int) -> None:
    for r, c in region:
        if grid.values[r][c] == 0:
            grid.values[r][c] = color
