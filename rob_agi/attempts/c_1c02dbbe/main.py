from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points.
    
    The function identifies seed points (non-gray, non-black cells), determines
    territories for each color, and expands these colors within their territories.
    The expansion is balanced and preserves the overall structure of the grid.
    
    Steps:
    1. Identify seed points
    2. Determine territories for each color
    3. Expand colors within their territories
    4. Fill remaining areas with gray
    5. Preserve the black border
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    # Step 1: Identify seed points
    seed_points = find_seed_points(input_grid)
    
    # Step 2: Determine territories
    territories = determine_territories(input_grid, seed_points)
    
    # Step 3 & 4: Expand colors and fill remaining areas
    output_grid = expand_colors(input_grid, seed_points, territories)
    
    # Step 5: Preserve black border
    preserve_border(output_grid)
    
    return output_grid

def find_seed_points(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    seed_points = {}
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in [0, 5]:  # Not black or gray
                if color not in seed_points:
                    seed_points[color] = []
                seed_points[color].append((r, c))
    return seed_points

def determine_territories(grid: ColoredGrid, seed_points: Dict[int, List[Tuple[int, int]]]) -> Dict[int, Tuple[int, int, int, int]]:
    territories = {}
    rows, cols = grid.get_dimensions()
    for color, points in seed_points.items():
        min_r = min(p[0] for p in points)
        max_r = max(p[0] for p in points)
        min_c = min(p[1] for p in points)
        max_c = max(p[1] for p in points)
        
        # Extend territory to edges or midpoints
        top = 0 if min_r == 1 else (min_r + max_r) // 2
        bottom = rows - 1 if max_r == rows - 2 else (min_r + max_r) // 2
        left = 0 if min_c == 1 else (min_c + max_c) // 2
        right = cols - 1 if max_c == cols - 2 else (min_c + max_c) // 2
        
        territories[color] = (top, left, bottom, right)
    return territories

def expand_colors(grid: ColoredGrid, seed_points: Dict[int, List[Tuple[int, int]]], territories: Dict[int, Tuple[int, int, int, int]]) -> ColoredGrid:
    output_grid = grid.deep_copy()
    for color, territory in territories.items():
        top, left, bottom, right = territory
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if output_grid.get_cell(r, c) == 5:  # Only replace gray cells
                    output_grid.set_cell(r, c, color)
    return output_grid

def preserve_border(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                grid.set_cell(r, c, 0)  # Set to black
