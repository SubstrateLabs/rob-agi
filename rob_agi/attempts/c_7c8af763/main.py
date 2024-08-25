from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set, Dict

def solve_7c8af763(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling regions bounded by gray lines with colors.
    
    The algorithm works as follows:
    1. Identifies regions bounded by gray (5) lines.
    2. For each region:
       a. Traces gray lines to find connected non-gray colors.
       b. Determines the fill color based on the traced colors and their positions.
    3. Fills each region while preserving original non-zero values.
    
    Returns a new ColoredGrid with the transformed values.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    
    for region in regions:
        fill_color = decide_fill_color(input_grid, region)
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

def decide_fill_color(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> int:
    colors = set(grid.values[r][c] for r, c in region if grid.values[r][c] not in [0, 5])
    if len(colors) == 1:
        return colors.pop()
    
    boundary = get_boundary_cells(region, grid)
    color_positions = trace_gray_lines(grid, boundary)
    
    if len(color_positions) == 1:
        return list(color_positions.keys())[0]
    elif 1 in color_positions and 2 in color_positions:
        return 1 if min(color_positions[1]) < min(color_positions[2]) else 2
    else:
        return min(color_positions.keys())

def get_boundary_cells(region: Set[Tuple[int, int]], grid: ColoredGrid) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    boundary = set()
    for r, c in region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 5:
                boundary.add((r, c))
                break
    return boundary

def trace_gray_lines(grid: ColoredGrid, start_cells: Set[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    queue = deque((r, c, 0) for r, c in start_cells)
    visited = set(start_cells)
    color_positions = {}
    
    while queue:
        r, c, dist = queue.popleft()
        if grid.values[r][c] not in [0, 5]:
            if grid.values[r][c] not in color_positions:
                color_positions[grid.values[r][c]] = []
            color_positions[grid.values[r][c]].append((r, c))
        else:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    queue.append((nr, nc, dist + 1))
                    visited.add((nr, nc))
    
    return color_positions

def fill_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color: int) -> None:
    for r, c in region:
        if grid.values[r][c] == 0:
            grid.values[r][c] = color
