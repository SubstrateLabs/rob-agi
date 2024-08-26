from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on structural features of shapes.
    
    The solution follows these steps:
    1. Identify connected regions of non-black cells.
    2. For each region:
       - Trace its contour and identify corner cells (yellow, 4).
       - Find intersection cells within the region (red, 2).
       - Process line segments between corners.
    3. Handle internal structures recursively.
    4. Preserve original colors for non-marked cells.
    5. Black cells (0) remain unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_connected_regions(input_grid)
    
    for region in regions:
        process_region(input_grid, output_grid, region)
    
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
                        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                queue.append((nr, nc))
                regions.append(region)
    return regions

def process_region(input_grid: ColoredGrid, output_grid: ColoredGrid, region: Set[Tuple[int, int]]):
    contour = trace_contour(input_grid, region)
    corners = detect_corners(contour)
    intersections = find_intersections(input_grid, region)
    
    for row, col in corners:
        output_grid.set_cell(row, col, 4)  # Yellow
    for row, col in intersections:
        output_grid.set_cell(row, col, 2)  # Red
    
    process_line_segments(input_grid, output_grid, corners)
    handle_internal_structures(input_grid, output_grid, region, set(contour))

def trace_contour(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    contour = []
    start = min(region)
    current = start
    directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    
    while True:
        contour.append(current)
        for dr, dc in directions:
            next_cell = (current[0] + dr, current[1] + dc)
            if next_cell in region and next_cell not in contour:
                current = next_cell
                break
        if current == start:
            break
    
    return contour

def detect_corners(contour: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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
        neighbors = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
                        if (row+dr, col+dc) in region)
        if neighbors >= 3:
            intersections.append((row, col))
    return intersections

def process_line_segments(input_grid: ColoredGrid, output_grid: ColoredGrid, corners: List[Tuple[int, int]]):
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i+1) % len(corners)]
        line = get_line(start, end)
        for row, col in line[1:-1]:  # Exclude start and end points
            if output_grid.get_cell(row, col) == 0:  # Only set if not already marked
                output_grid.set_cell(row, col, input_grid.get_cell(row, col))

def get_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    line = []
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        line.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return line

def handle_internal_structures(input_grid: ColoredGrid, output_grid: ColoredGrid, region: Set[Tuple[int, int]], contour: Set[Tuple[int, int]]):
    internal_cells = region - contour
    if internal_cells:
        internal_regions = find_connected_regions(ColoredGrid(values=[[1 if (r, c) in internal_cells else 0 for c in range(input_grid.num_cols)] for r in range(input_grid.num_rows)]))
        for internal_region in internal_regions:
            process_region(input_grid, output_grid, internal_region)
