from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_2c0b0aff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2c0b0aff challenge by identifying and extracting the fundamental repeating pattern from the input grid.
    
    The function performs the following steps:
    1. Identifies non-black regions in the input grid
    2. Creates a bounding box around all non-black cells
    3. Analyzes the structure of the pattern for symmetries and recurring sub-patterns
    4. Extracts potential core patterns of various sizes
    5. Optimizes the pattern by removing redundant parts
    6. Verifies and refines the pattern
    7. Returns the optimized pattern as a compact ColoredGrid
    
    Args:
    input_grid (ColoredGrid): The input grid containing partial pattern information
    
    Returns:
    ColoredGrid: A compact grid containing the extracted fundamental repeating pattern
    """
    # Step 1: Identify non-black regions
    non_black_cells = find_non_black_cells(input_grid)
    
    if not non_black_cells:
        return ColoredGrid(values=[[]])
    
    # Step 2: Create bounding box
    min_r, min_c, max_r, max_c = get_bounding_box(non_black_cells)
    
    # Step 3 & 4: Analyze structure and extract potential core patterns
    pattern = find_optimal_pattern(input_grid, min_r, min_c, max_r, max_c)
    
    # Step 5 & 6: Optimize, verify, and refine the pattern
    optimized_pattern = optimize_pattern(pattern)
    
    # Step 7: Return the optimized pattern
    return optimized_pattern

def find_non_black_cells(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) != 0}

def get_bounding_box(cells: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return min_r, min_c, max_r, max_c

def find_optimal_pattern(grid: ColoredGrid, min_r: int, min_c: int, max_r: int, max_c: int) -> ColoredGrid:
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    best_pattern = None
    best_score = float('inf')
    
    for pattern_height in range(1, height + 1):
        for pattern_width in range(1, width + 1):
            pattern = grid.extract_subgrid(min_r, min_c, pattern_height, pattern_width)
            score = pattern_score(grid, pattern, min_r, min_c, max_r, max_c)
            if score < best_score:
                best_score = score
                best_pattern = pattern
    
    return best_pattern

def pattern_score(grid: ColoredGrid, pattern: ColoredGrid, min_r: int, min_c: int, max_r: int, max_c: int) -> float:
    p_rows, p_cols = pattern.get_dimensions()
    mismatches = 0
    total_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid.get_cell(r, c) != pattern.get_cell((r - min_r) % p_rows, (c - min_c) % p_cols):
                mismatches += 1
    
    return mismatches / total_cells + (p_rows * p_cols) / 100  # Add pattern size penalty

def optimize_pattern(pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = pattern.get_dimensions()
    
    # Remove black rows and columns from the edges
    top = next(r for r in range(rows) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
    bottom = next(r for r in range(rows - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for c in range(cols)))
    left = next(c for c in range(cols) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
    right = next(c for c in range(cols - 1, -1, -1) if any(pattern.get_cell(r, c) != 0 for r in range(rows)))
    
    return pattern.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
