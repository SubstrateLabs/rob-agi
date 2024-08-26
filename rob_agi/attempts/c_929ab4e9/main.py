from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the central region to be filled
    2. Analyzing the surrounding area for patterns
    3. Creating a pattern map based on relative positions
    4. Filling the central region using the pattern map
    5. Returning the modified grid
    """
    # Identify the central region
    central_region = identify_central_region(input_grid)
    
    # Analyze surrounding area and create pattern map
    pattern_map = create_pattern_map(input_grid, central_region)
    
    # Fill the central region
    output_grid = fill_central_region(input_grid, central_region, pattern_map)
    
    return output_grid

def identify_central_region(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    central_region = []
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2:  # Assuming 2 (red) is always the color to be replaced
                central_region.append((r, c))
    return central_region

def create_pattern_map(grid: ColoredGrid, central_region: List[Tuple[int, int]]) -> Dict[Tuple[float, float], int]:
    pattern_map = {}
    rows, cols = grid.get_dimensions()
    min_r = min(r for r, _ in central_region)
    max_r = max(r for r, _ in central_region)
    min_c = min(c for _, c in central_region)
    max_c = max(c for _, c in central_region)
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in central_region:
                rel_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.5
                rel_c = (c - min_c) / (max_c - min_c) if max_c > min_c else 0.5
                pattern_map[(rel_r, rel_c)] = grid.values[r][c]
    
    return pattern_map

def fill_central_region(grid: ColoredGrid, central_region: List[Tuple[int, int]], pattern_map: Dict[Tuple[float, float], int]) -> ColoredGrid:
    output_grid = grid.deep_copy()
    min_r = min(r for r, _ in central_region)
    max_r = max(r for r, _ in central_region)
    min_c = min(c for _, c in central_region)
    max_c = max(c for _, c in central_region)
    
    for r, c in central_region:
        rel_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.5
        rel_c = (c - min_c) / (max_c - min_c) if max_c > min_c else 0.5
        
        # Find the closest pattern point
        closest_point = min(pattern_map.keys(), key=lambda k: ((k[0]-rel_r)**2 + (k[1]-rel_c)**2))
        output_grid.values[r][c] = pattern_map[closest_point]
    
    return output_grid
