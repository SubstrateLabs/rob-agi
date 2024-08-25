from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set

def solve_7c8af763(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling regions bounded by gray lines with colors.
    
    The algorithm works as follows:
    1. Identifies regions bounded by gray (5) lines.
    2. For each region:
       a. If it contains a single non-gray color, fills with that color.
       b. Otherwise, finds the nearest non-gray color outside the region.
       c. If multiple equidistant colors are found, chooses based on grid position.
    3. Fills each region while preserving original non-zero values.
    
    Returns a new ColoredGrid with the transformed values.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    
    for region in regions:
        fill_color = decide_fill_color(input_grid, region, rows, cols)
        fill_region(output_grid, region, fill_color)
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 5:
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
            if 0 <= nr < rows and 0 <= nc < cols:
                queue.append((nr, nc))
    
    return region

def decide_fill_color(grid: ColoredGrid, region: Set[Tuple[int, int]], rows: int, cols: int) -> int:
    colors = set(grid.values[r][c] for r, c in region if grid.values[r][c] not in [0, 5])
    if len(colors) == 1:
        return colors.pop()
    
    boundary = set((r, c) for r, c in region if any(0 <= r+dr < rows and 0 <= c+dc < cols and (r+dr, c+dc) not in region for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]))
    nearest_colors = find_nearest_colors(grid, boundary, region)
    
    if len(nearest_colors) == 1:
        return nearest_colors[0]
    elif 1 in nearest_colors and 2 in nearest_colors:
        return 1 if sum(r for r, _ in region) / len(region) < rows / 2 else 2
    else:
        return min(nearest_colors)

def find_nearest_colors(grid: ColoredGrid, boundary: Set[Tuple[int, int]], region: Set[Tuple[int, int]]) -> List[int]:
    rows, cols = grid.get_dimensions()
    queue = deque((r, c, 0) for r, c in boundary)
    visited = set(region)
    nearest_colors = []
    nearest_distance = float('inf')
    
    while queue:
        r, c, dist = queue.popleft()
        if dist > nearest_distance:
            break
        if grid.values[r][c] not in [0, 5]:
            if dist < nearest_distance:
                nearest_colors = [grid.values[r][c]]
                nearest_distance = dist
            elif dist == nearest_distance:
                nearest_colors.append(grid.values[r][c])
        else:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    queue.append((nr, nc, dist + 1))
                    visited.add((nr, nc))
    
    return nearest_colors

def fill_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color: int) -> None:
    for r, c in region:
        if grid.values[r][c] == 0:
            grid.values[r][c] = color
