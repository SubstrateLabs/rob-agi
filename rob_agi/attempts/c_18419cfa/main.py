from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_18419cfa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 18419cfa challenge by expanding red (2) patterns within sky blue (8) regions.
    
    The function identifies connected sky blue regions, finds red pixels within them,
    and expands the red patterns symmetrically. It handles various patterns including
    single pixels, L-shapes, rectangles, crosses, and other complex shapes. The expansion
    is done to maximize symmetry and fill available space within the sky blue boundaries.
    """
    grid = input_grid.deep_copy()
    sky_blue_regions = find_connected_regions(grid, 8)
    
    for region in sky_blue_regions:
        red_pixels = set(coord for coord in region if grid.get_cell(coord[0], coord[1]) == 2)
        expanded_red_pixels = expand_red_patterns(grid, region, red_pixels)
        
        for x, y in expanded_red_pixels:
            grid.set_cell(x, y, 2)
    
    return grid

def find_connected_regions(grid: ColoredGrid, color: int) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r: int, c: int) -> Set[Tuple[int, int]]:
        stack = [(r, c)]
        region = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and grid.get_cell(r, c) == color:
                visited.add((r, c))
                region.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) == color:
                regions.append(dfs(r, c))

    return regions

def expand_red_patterns(grid: ColoredGrid, region: Set[Tuple[int, int]], red_pixels: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    expanded_pixels = set(red_pixels)
    region_bounds = get_region_bounds(region)
    
    while True:
        new_pixels = expand_step(expanded_pixels, region, region_bounds)
        if not new_pixels:
            break
        expanded_pixels.update(new_pixels)
    
    return expanded_pixels

def get_region_bounds(region: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return min_r, max_r, min_c, max_c

def expand_step(pixels: Set[Tuple[int, int]], region: Set[Tuple[int, int]], bounds: Tuple[int, int, int, int]) -> Set[Tuple[int, int]]:
    min_r, max_r, min_c, max_c = bounds
    new_pixels = set()
    
    for r, c in pixels:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if min_r <= nr <= max_r and min_c <= nc <= max_c and (nr, nc) in region and (nr, nc) not in pixels:
                new_pixels.add((nr, nc))
    
    return new_pixels
