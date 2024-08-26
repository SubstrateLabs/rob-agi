from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing blue shapes with thin sections to red,
    while preserving blue shapes that are at least two cells wide throughout. The function
    identifies connected blue regions, checks for thin sections, and changes the color of
    shapes with thin sections to red. Original red shapes and black cells remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with blue shapes containing thin sections changed to red.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = find_blue_regions(output_grid)
    
    for region in blue_regions:
        if has_thin_section(output_grid, region):
            for r, c in region:
                output_grid.set_cell(r, c, 2)  # Change to red
    
    return output_grid

def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r, c):
        if (r, c) in visited or grid.get_cell(r, c) != 1:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                region.extend(dfs(nr, nc))
        return region

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1 and (r, c) not in visited:
                regions.append(dfs(r, c))

    return regions

def has_thin_section(grid: ColoredGrid, region: List[Tuple[int, int]]) -> bool:
    region_set = set(region)
    for r, c in region:
        adjacent_blue = sum(
            1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if (r + dr, c + dc) in region_set
        )
        if adjacent_blue < 2:
            return True
    return False
