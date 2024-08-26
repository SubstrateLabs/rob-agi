from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1a6449f1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a significant subgrid from the input grid based on the following steps:
    1. Identify the most significant shapes/patterns in the grid.
    2. For each significant shape, find potential subgrids that capture key features.
    3. Score each potential subgrid based on size, shape capture, and position.
    4. Select and return the subgrid with the highest score.
    """
    rows, cols = input_grid.get_dimensions()
    color_frequencies = input_grid.get_color_frequencies()
    sorted_colors = sorted(color_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    best_subgrid = None
    best_score = -1

    for color, _ in sorted_colors:
        if color == 0:  # Skip black
            continue
        
        regions = find_connected_regions(input_grid, color)
        for region in regions:
            subgrid = find_best_subgrid(input_grid, region, color)
            if subgrid:
                score = score_subgrid(subgrid, input_grid, color)
                if score > best_score:
                    best_score = score
                    best_subgrid = subgrid

    return best_subgrid if best_subgrid else ColoredGrid(values=[[0]])

def find_connected_regions(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(color)

def find_best_subgrid(grid: ColoredGrid, region: List[Tuple[int, int]], color: int) -> ColoredGrid:
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    
    best_subgrid = None
    best_ratio = 0
    
    for top in range(min_row, max_row + 1):
        for left in range(min_col, max_col + 1):
            for bottom in range(top, max_row + 1):
                for right in range(left, max_col + 1):
                    subgrid = grid.extract_subgrid(top, left, bottom-top+1, right-left+1)
                    ratio = sum(row.count(color) for row in subgrid.values) / ((bottom-top+1) * (right-left+1))
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_subgrid = subgrid
    
    return best_subgrid

def score_subgrid(subgrid: ColoredGrid, original_grid: ColoredGrid, color: int) -> float:
    subgrid_rows, subgrid_cols = subgrid.get_dimensions()
    original_rows, original_cols = original_grid.get_dimensions()
    
    size_score = (subgrid_rows * subgrid_cols) / (original_rows * original_cols)
    color_ratio = sum(row.count(color) for row in subgrid.values) / (subgrid_rows * subgrid_cols)
    
    return size_score * color_ratio
