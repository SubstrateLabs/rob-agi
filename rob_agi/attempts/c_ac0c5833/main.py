from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
import random

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions around yellow (4) cells,
    replicating patterns, and ensuring connectivity. The function:
    1. Identifies yellow cells and existing red structures.
    2. Expands 3x3 red regions around yellow cells, leaving one corner empty.
    3. Connects nearby red expansions and structures.
    4. Replicates recognized red patterns relative to yellow cells.
    5. Ensures overall connectivity of red regions.
    6. Cleans up isolated red cells and verifies yellow cell neighborhoods.
    7. Performs final checks to maintain pattern integrity.
    """
    grid = input_grid.deep_copy()
    yellow_cells = find_colored_cells(grid, 4)
    red_cells = find_colored_cells(grid, 2)
    
    for row, col in yellow_cells:
        expand_around_yellow(grid, row, col)
    
    connect_expansions(grid)
    replicate_patterns(grid, yellow_cells, red_cells)
    ensure_connectivity(grid)
    clean_up(grid, yellow_cells)
    final_check(grid, yellow_cells)
    
    return grid

def find_colored_cells(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == color]

def expand_around_yellow(grid: ColoredGrid, row: int, col: int):
    corners = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
    empty_corner = choose_empty_corner(grid, corners)
    
    for i in range(max(0, row-1), min(grid.num_rows, row+2)):
        for j in range(max(0, col-1), min(grid.num_cols, col+2)):
            if (i, j) != (row, col) and (i, j) != empty_corner and grid.values[i][j] == 0:
                grid.values[i][j] = 2

def choose_empty_corner(grid: ColoredGrid, corners: List[Tuple[int, int]]) -> Tuple[int, int]:
    valid_corners = [corner for corner in corners if is_valid_position(grid, corner[0], corner[1])]
    black_corners = [corner for corner in valid_corners if grid.values[corner[0]][corner[1]] == 0]
    
    if black_corners:
        return max(black_corners, key=lambda c: count_black_neighbors(grid, c[0], c[1]))
    elif valid_corners:
        return random.choice(valid_corners)
    else:
        return corners[0]  # Default to top-left if no valid corners

def is_valid_position(grid: ColoredGrid, row: int, col: int) -> bool:
    return 0 <= row < grid.num_rows and 0 <= col < grid.num_cols

def count_black_neighbors(grid: ColoredGrid, row: int, col: int) -> int:
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if is_valid_position(grid, nr, nc) and grid.values[nr][nc] == 0:
                count += 1
    return count

def connect_expansions(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2:
                connect_neighbors(grid, r, c)

def connect_neighbors(grid: ColoredGrid, row: int, col: int):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if is_valid_position(grid, nr, nc) and grid.values[nr][nc] == 0:
            if not creates_2x2_square(grid, nr, nc):
                grid.values[nr][nc] = 2

def creates_2x2_square(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr in [-1, 0]:
        for dc in [-1, 0]:
            if all(grid.values[row+r][col+c] == 2 for r in [dr, dr+1] for c in [dc, dc+1] if is_valid_position(grid, row+r, col+c)):
                return True
    return False

def replicate_patterns(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]], red_cells: List[Tuple[int, int]]):
    patterns = recognize_patterns(grid, yellow_cells, red_cells)
    for yellow_cell in yellow_cells:
        for pattern in patterns:
            apply_pattern(grid, yellow_cell, pattern)

def recognize_patterns(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]], red_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    patterns = []
    for yellow_row, yellow_col in yellow_cells:
        pattern = []
        for red_row, red_col in red_cells:
            if abs(red_row - yellow_row) <= 3 and abs(red_col - yellow_col) <= 3:
                pattern.append((red_row - yellow_row, red_col - yellow_col))
        if pattern:
            patterns.append(pattern)
    return patterns

def apply_pattern(grid: ColoredGrid, yellow_cell: Tuple[int, int], pattern: List[Tuple[int, int]]):
    yellow_row, yellow_col = yellow_cell
    for dr, dc in pattern:
        new_row, new_col = yellow_row + dr, yellow_col + dc
        if is_valid_position(grid, new_row, new_col) and grid.values[new_row][new_col] == 0:
            if not creates_2x2_square(grid, new_row, new_col):
                grid.values[new_row][new_col] = 2

def ensure_connectivity(grid: ColoredGrid):
    red_regions = find_red_regions(grid)
    if len(red_regions) > 1:
        connect_red_regions(grid, red_regions)

def find_red_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    visited = set()
    regions = []
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and (r, c) not in visited:
                region = flood_fill(grid, r, c, visited)
                regions.append(region)
    return regions

def flood_fill(grid: ColoredGrid, row: int, col: int, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    stack = [(row, col)]
    region = set()
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and grid.values[r][c] == 2:
            visited.add((r, c))
            region.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid_position(grid, nr, nc):
                    stack.append((nr, nc))
    return region

def connect_red_regions(grid: ColoredGrid, regions: List[Set[Tuple[int, int]]]):
    while len(regions) > 1:
        region1, region2 = regions[:2]
        path = find_shortest_path(grid, region1, region2)
        for r, c in path:
            if grid.values[r][c] == 0:
                grid.values[r][c] = 2
        new_region = region1.union(region2).union(set(path))
        regions = [new_region] + regions[2:]

def find_shortest_path(grid: ColoredGrid, region1: Set[Tuple[int, int]], region2: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    start = next(iter(region1))
    end = min(region2, key=lambda x: manhattan_distance(start, x))
    queue = [(start, [start])]
    visited = set()
    
    while queue:
        (r, c), path = queue.pop(0)
        if (r, c) == end:
            return path
        if (r, c) not in visited:
            visited.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid_position(grid, nr, nc) and (nr, nc) not in visited:
                    new_path = path + [(nr, nc)]
                    queue.append(((nr, nc), new_path))
    return []

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def clean_up(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]]):
    remove_isolated_reds(grid)
    ensure_yellow_neighbors(grid, yellow_cells)

def remove_isolated_reds(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and is_isolated(grid, r, c):
                grid.values[r][c] = 0

def is_isolated(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = row + dr, col + dc
        if is_valid_position(grid, nr, nc) and grid.values[nr][nc] in [2, 4]:
            return False
    return True

def ensure_yellow_neighbors(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]]):
    for row, col in yellow_cells:
        if not any(grid.values[row+dr][col+dc] == 2 
                   for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] 
                   if is_valid_position(grid, row+dr, col+dc)):
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if is_valid_position(grid, nr, nc) and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = 2
                    break

def final_check(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]]):
    for row, col in yellow_cells:
        ensure_3x3_pattern(grid, row, col)

def ensure_3x3_pattern(grid: ColoredGrid, row: int, col: int):
    corners = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
    if all(is_valid_position(grid, r, c) and grid.values[r][c] == 2 for r, c in corners):
        empty_corner = choose_empty_corner(grid, corners)
        grid.values[empty_corner[0]][empty_corner[1]] = 0
