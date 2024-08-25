from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing unique blue shapes to red
    while preserving repeated blue shapes. The function identifies connected regions
    of blue cells, generates a unique identifier for each shape, and then changes
    the color of unique shapes to red.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with unique blue shapes changed to red.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = find_connected_regions(output_grid, 1)  # 1 represents blue
    
    shape_counts = defaultdict(int)
    for region in blue_regions:
        shape_id = generate_shape_identifier(region)
        shape_counts[shape_id] += 1
    
    for region in blue_regions:
        shape_id = generate_shape_identifier(region)
        if shape_counts[shape_id] == 1:
            for r, c in region:
                output_grid.set_cell(r, c, 2)  # 2 represents red
    
    return output_grid

def find_connected_regions(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected regions of a specific color in the grid."""
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                regions.append(region)
    
    return regions

def generate_shape_identifier(region: List[Tuple[int, int]]) -> str:
    """Generate a unique identifier for a shape based on its relative coordinates."""
    if not region:
        return ""
    
    min_r = min(r for r, _ in region)
    min_c = min(c for _, c in region)
    normalized = [(r - min_r, c - min_c) for r, c in region]
    normalized.sort()
    
    return ",".join(f"{r},{c}" for r, c in normalized)
