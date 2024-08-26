from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def extract_color_key(grid):
    color_key = []
    for row in grid:
        for cell in row:
            if cell != 0 and cell != 5 and cell not in color_key:
                color_key.append(cell)
    return color_key

def get_sorted_regions(grid):
    regions = []
    visited = set()
    
    def flood_fill(r, c, color):
        if (r, c) in visited or r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(flood_fill(r + dr, c + dc, color))
        return region
    
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell not in [0, 5] and (r, c) not in visited:
                region = flood_fill(r, c, cell)
                regions.append(region)
    
    return sorted(regions, key=lambda r: (min(r)[0], min(r)[1]))

def transform_grid(input_grid):
    color_key = extract_color_key(input_grid)
    sorted_regions = get_sorted_regions(input_grid)
    
    new_grid = [[0 for _ in row] for row in input_grid]
    
    # Copy gray cells
    for r, row in enumerate(input_grid):
        for c, cell in enumerate(row):
            if cell == 5:
                new_grid[r][c] = 5
    
    # Assign new colors to regions
    for i, region in enumerate(sorted_regions):
        new_color = color_key[i % len(color_key)]
        for r, c in region:
            new_grid[r][c] = new_color
    
    return new_grid

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Extracting the ordered list of colors from the non-gray, non-black cells in the gray area.
    2. Identifying and sorting non-gray, non-black regions in the grid based on their top-left coordinate.
    3. Assigning new colors to the sorted regions cyclically using the extracted color key.
    4. Creating a new grid with the transformed colors while preserving the gray area.
    5. Returning the transformed grid.
    """
    transformed_values = transform_grid(input_grid.values)
    return ColoredGrid(values=transformed_values)
