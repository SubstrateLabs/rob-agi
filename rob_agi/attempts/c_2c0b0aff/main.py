from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_2c0b0aff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2c0b0aff challenge by identifying and extracting the fundamental repeating pattern from the input grid.
    
    The function performs the following steps:
    1. Identifies non-black regions in the input grid
    2. Generates pattern candidates of various sizes
    3. Scores each pattern candidate based on its occurrence in the input grid
    4. Selects the best pattern that explains all non-black regions
    5. Optimizes the pattern to ensure it's the smallest repeating unit
    6. Returns the optimized pattern as a compact ColoredGrid
    
    Args:
    input_grid (ColoredGrid): The input grid containing partial pattern information
    
    Returns:
    ColoredGrid: A compact grid containing the extracted fundamental repeating pattern
    """
    # Step 1: Identify non-black regions
    non_black_cells = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.get_cell(r, c) != 0]
    
    if not non_black_cells:
        return ColoredGrid(values=[[]])
    
    # Step 2 & 3: Generate and score pattern candidates
    best_pattern, best_score = find_best_pattern(input_grid, non_black_cells)
    
    # Step 4 & 5: Optimize the pattern
    optimized_pattern = optimize_pattern(best_pattern)
    
    # Step 6: Return the optimized pattern
    return optimized_pattern

def find_best_pattern(grid: ColoredGrid, non_black_cells: List[Tuple[int, int]]) -> Tuple[ColoredGrid, int]:
    min_r = min(r for r, _ in non_black_cells)
    min_c = min(c for _, c in non_black_cells)
    max_r = max(r for r, _ in non_black_cells)
    max_c = max(c for _, c in non_black_cells)
    
    best_pattern = None
    best_score = -1
    
    for height in range(1, max_r - min_r + 2):
        for width in range(1, max_c - min_c + 2):
            for start_r in range(min_r, max_r - height + 2):
                for start_c in range(min_c, max_c - width + 2):
                    pattern = grid.extract_subgrid(start_r, start_c, height, width)
                    score = score_pattern(grid, pattern)
                    if score > best_score or (score == best_score and pattern.num_rows * pattern.num_cols < best_pattern.num_rows * best_pattern.num_cols):
                        best_pattern = pattern
                        best_score = score
    
    return best_pattern, best_score

def score_pattern(grid: ColoredGrid, pattern: ColoredGrid) -> int:
    score = 0
    rows, cols = grid.get_dimensions()
    p_rows, p_cols = pattern.get_dimensions()
    
    for r in range(rows - p_rows + 1):
        for c in range(cols - p_cols + 1):
            match_score = sum(
                grid.get_cell(r+i, c+j) == pattern.get_cell(i, j)
                for i in range(p_rows)
                for j in range(p_cols)
                if grid.get_cell(r+i, c+j) != 0 or pattern.get_cell(i, j) != 0
            )
            score += match_score / (p_rows * p_cols)
    
    return score

def optimize_pattern(pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = pattern.get_dimensions()
    
    # Check if pattern can be reduced horizontally
    for width in range(1, cols):
        if cols % width == 0:
            if all(pattern.get_cell(r, c) == pattern.get_cell(r, c % width) for r in range(rows) for c in range(cols)):
                return optimize_pattern(pattern.extract_subgrid(0, 0, rows, width))
    
    # Check if pattern can be reduced vertically
    for height in range(1, rows):
        if rows % height == 0:
            if all(pattern.get_cell(r, c) == pattern.get_cell(r % height, c) for r in range(rows) for c in range(cols)):
                return optimize_pattern(pattern.extract_subgrid(0, 0, height, cols))
    
    return pattern
