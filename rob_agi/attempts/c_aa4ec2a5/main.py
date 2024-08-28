from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_aa4ec2a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Preserves the yellow (4) background where no blue regions exist.
    2. Identifies and classifies blue (1) regions based on size.
    3. Processes large regions (>9 cells): outlines with red (2), fills with sky blue (8), and adds internal structure.
    4. Processes medium regions (4-9 cells): outlines with red (2), fills with blue (1).
    5. Processes small regions (1-3 cells): creates special structures with red (2) outline and blue (1) or sky blue (8) fill.
    6. Adds a red (2) border around transformed regions and at grid edges.
    7. Creates a subtle grid structure with red (2) lines every 5 cells in untransformed areas.
    8. Adjusts intersections and corners to ensure proper connectivity of red lines.
    9. Performs a final pass to ensure consistency and proper integration of all elements.
    """
    new_grid = input_grid.deep_copy()
    blue_regions = find_blue_regions(new_grid)
    
    for region in blue_regions:
        if len(region) > 9:
            process_large_region(new_grid, region)
        elif 4 <= len(region) <= 9:
            process_medium_region(new_grid, region)
        else:
            process_small_region(new_grid, region)
    
    create_subtle_grid_structure(new_grid)
    add_border(new_grid)
    adjust_intersections_and_corners(new_grid)
    final_pass(new_grid)
    
    return new_grid

def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    visited = set()
    regions = []
    for y in range(grid.num_rows):
        for x in range(grid.num_cols):
            if grid.values[y][x] == 1 and (y, x) not in visited:
                region = []
                queue = deque([(y, x)])
                while queue:
                    cy, cx = queue.popleft()
                    if (cy, cx) not in visited and grid.values[cy][cx] == 1:
                        visited.add((cy, cx))
                        region.append((cy, cx))
                        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols:
                                queue.append((ny, nx))
                regions.append(region)
    return regions

def create_subtle_grid_structure(grid: ColoredGrid):
    for y in range(0, grid.num_rows, 5):
        for x in range(grid.num_cols):
            if grid.values[y][x] == 4:
                grid.values[y][x] = 2
    for x in range(0, grid.num_cols, 5):
        for y in range(grid.num_rows):
            if grid.values[y][x] == 4:
                grid.values[y][x] = 2

def add_border(grid: ColoredGrid):
    for y in range(grid.num_rows):
        if grid.values[y][0] == 4:
            grid.values[y][0] = 2
        if grid.values[y][-1] == 4:
            grid.values[y][-1] = 2
    for x in range(grid.num_cols):
        if grid.values[0][x] == 4:
            grid.values[0][x] = 2
        if grid.values[-1][x] == 4:
            grid.values[-1][x] = 2

def process_large_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    outline_region(grid, region, 2)
    fill_region(grid, region, 8)
    add_internal_structure(grid, region)

def process_medium_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    outline_region(grid, region, 2)
    fill_region(grid, region, 1)

def process_small_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    if len(region) == 1:
        y, x = region[0]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols:
                    grid.values[ny][nx] = 2 if (dy, dx) != (0, 0) else 8
    else:
        outline_region(grid, region, 2)
        fill_region(grid, region, 1)

def outline_region(grid: ColoredGrid, region: List[Tuple[int, int]], color: int):
    for y, x in region:
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols and (ny, nx) not in region:
                grid.values[ny][nx] = color

def fill_region(grid: ColoredGrid, region: List[Tuple[int, int]], color: int):
    for y, x in region:
        grid.values[y][x] = color

def add_internal_structure(grid: ColoredGrid, region: List[Tuple[int, int]]):
    min_y, min_x = min(y for y, _ in region), min(x for _, x in region)
    max_y, max_x = max(y for y, _ in region), max(x for _, x in region)
    mid_y, mid_x = (min_y + max_y) // 2, (min_x + max_x) // 2
    
    for y in range(min_y, max_y + 1):
        if min_x < mid_x < max_x and (y, mid_x) in region:
            grid.values[y][mid_x] = 2
    for x in range(min_x, max_x + 1):
        if min_y < mid_y < max_y and (mid_y, x) in region:
            grid.values[mid_y][x] = 2
    
    # Add magenta square if the region is T-shaped
    if is_t_shaped(region):
        for dy in range(2):
            for dx in range(2):
                ny, nx = min_y + dy, min_x + dx
                if (ny, nx) in region:
                    grid.values[ny][nx] = 6

def is_t_shaped(region: List[Tuple[int, int]]) -> bool:
    min_y, min_x = min(y for y, _ in region), min(x for _, x in region)
    max_y, max_x = max(y for y, _ in region), max(x for _, x in region)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    if width > height and height >= 3:
        top_row = sum(1 for x in range(min_x, max_x + 1) if (min_y, x) in region)
        stem = all((y, (min_x + max_x) // 2) in region for y in range(min_y, max_y + 1))
        return top_row == width and stem
    return False

def adjust_intersections_and_corners(grid: ColoredGrid):
    for y in range(1, grid.num_rows - 1):
        for x in range(1, grid.num_cols - 1):
            if grid.values[y][x] == 2:
                neighbors = sum(1 for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)] if grid.values[y+dy][x+dx] == 2)
                if neighbors >= 2:
                    for dy, dx in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        if 0 <= y+dy < grid.num_rows and 0 <= x+dx < grid.num_cols:
                            grid.values[y+dy][x+dx] = 2

def final_pass(grid: ColoredGrid):
    for y in range(grid.num_rows):
        for x in range(grid.num_cols):
            if grid.values[y][x] == 1 and is_part_of_large_region(grid, y, x):
                grid.values[y][x] = 8

def is_part_of_large_region(grid: ColoredGrid, y: int, x: int) -> bool:
    visited = set()
    queue = deque([(y, x)])
    region_size = 0
    while queue:
        cy, cx = queue.popleft()
        if (cy, cx) not in visited and grid.values[cy][cx] in [1, 8]:
            visited.add((cy, cx))
            region_size += 1
            if region_size > 9:
                return True
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols:
                    queue.append((ny, nx))
    return False
