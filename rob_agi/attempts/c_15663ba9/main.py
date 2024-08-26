from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on structural features of shapes.
    
    The solution follows these steps:
    1. Identify connected regions of non-black cells.
    2. For each region:
       - Trace its contour and identify potential corner/endpoint cells (yellow, 4).
       - Find potential intersection cells within the region (red, 2).
       - Apply markers based on the shape's structure.
    3. Preserve the original color for non-marked cells.
    4. Black cells (0) remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_connected_regions(input_grid)
    
    for region in regions:
        contour = trace_contour(input_grid, region)
        corners = detect_corners(input_grid, contour)
        intersections = find_intersections(input_grid, region)
        
        for row, col in corners:
            output_grid.set_cell(row, col, 4)  # Yellow
        for row, col in intersections:
            output_grid.set_cell(row, col, 2)  # Red
    
    return output_grid

def find_connected_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    visited = set()
    regions = []
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if grid.get_cell(row, col) != 0 and (row, col) not in visited:
                region = set()
                queue = deque([(row, col)])
                while queue:
                    r, c = queue.popleft()
                    if (r, c) not in visited and grid.get_cell(r, c) != 0:
                        visited.add((r, c))
                        region.add((r, c))
                        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                queue.append((nr, nc))
                regions.append(region)
    return regions

def trace_contour(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    contour = []
    start = next(iter(region))
    current = start
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    dir_index = 0
    
    while True:
        contour.append(current)
        for _ in range(4):
            next_cell = (current[0] + directions[dir_index][0], current[1] + directions[dir_index][1])
            if next_cell in region:
                current = next_cell
                dir_index = (dir_index - 1) % 4
                break
            dir_index = (dir_index + 1) % 4
        if current == start:
            break
    
    return contour

def detect_corners(grid: ColoredGrid, contour: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    corners = []
    for i in range(len(contour)):
        prev = contour[i-1]
        curr = contour[i]
        next = contour[(i+1) % len(contour)]
        if (prev[0] - curr[0], prev[1] - curr[1]) != (curr[0] - next[0], curr[1] - next[1]):
            corners.append(curr)
    return corners

def find_intersections(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    intersections = []
    for row, col in region:
        neighbors = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                        if (row+dr, col+dc) in region)
        if neighbors >= 3:
            intersections.append((row, col))
    return intersections
