from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_2c0b0aff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2c0b0aff challenge by identifying and extracting the fundamental repeating pattern from the input grid.
    
    The function performs the following steps:
    1. Identifies non-black regions in the input grid
    2. Extracts the largest connected non-black region
    3. Finds the smallest repeating pattern within this region
    4. Optimizes the pattern by removing any redundant parts
    5. Returns the optimized pattern as a compact ColoredGrid
    
    Args:
    input_grid (ColoredGrid): The input grid containing partial pattern information
    
    Returns:
    ColoredGrid: A compact grid containing the extracted fundamental repeating pattern
    """
    # Step 1: Identify non-black regions
    non_black_cells = find_non_black_cells(input_grid)
    
    if not non_black_cells:
        return ColoredGrid(values=[[]])
    
    # Step 2: Extract largest connected region
    largest_region = find_largest_connected_region(input_grid, non_black_cells)
    
    # Step 3 & 4: Find and optimize the pattern
    pattern = find_smallest_repeating_pattern(input_grid, largest_region)
    
    # Step 5: Return the optimized pattern
    return pattern

def find_non_black_cells(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) != 0}

def find_largest_connected_region(grid: ColoredGrid, non_black_cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    largest_region = set()
    for cell in non_black_cells:
        if cell not in largest_region:
            region = bfs_region(grid, cell)
            if len(region) > len(largest_region):
                largest_region = region
    return largest_region

def bfs_region(grid: ColoredGrid, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    queue = deque([start])
    region = set()
    while queue:
        r, c = queue.popleft()
        if (r, c) not in region and grid.get_cell(r, c) != 0:
            region.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    queue.append((nr, nc))
    return region

def find_smallest_repeating_pattern(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> ColoredGrid:
    min_r = min(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_r = max(r for r, _ in region)
    max_c = max(c for _, c in region)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    for pattern_height in range(1, height + 1):
        for pattern_width in range(1, width + 1):
            pattern = grid.extract_subgrid(min_r, min_c, pattern_height, pattern_width)
            if is_valid_pattern(grid, pattern, region):
                return optimize_pattern(pattern)
    
    # If no pattern found, return the entire region
    return grid.extract_subgrid(min_r, min_c, height, width)

def is_valid_pattern(grid: ColoredGrid, pattern: ColoredGrid, region: Set[Tuple[int, int]]) -> bool:
    p_rows, p_cols = pattern.get_dimensions()
    for r, c in region:
        if grid.get_cell(r, c) != pattern.get_cell(r % p_rows, c % p_cols):
            return False
    return True

def optimize_pattern(pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = pattern.get_dimensions()
    
    # Remove black rows and columns from the edges
    top = next(r for r in range(rows) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
    bottom = next(r for r in range(rows - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
    left = next(c for c in range(cols) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
    right = next(c for c in range(cols - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
    
    return pattern.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
