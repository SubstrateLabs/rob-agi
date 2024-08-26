from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def extract_color_key(grid):
    color_key = []
    for row in grid:
        for cell in row:
            if cell == 5:  # Found the gray area
                for c in row:
                    if c != 0 and c != 5 and c not in color_key:
                        color_key.append(c)
                return color_key
    return color_key

def get_sorted_regions(grid):
    regions = []
    visited = set()
    rows, cols = len(grid), len(grid[0])
    
    def flood_fill(r, c, color):
        queue = deque([(r, c)])
        region = []
        while queue:
            r, c = queue.popleft()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid[r][c] == color:
                visited.add((r, c))
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((r + dr, c + dc))
        return region
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [0, 5] and (r, c) not in visited:
                region = flood_fill(r, c, grid[r][c])
                regions.append(region)
    
    return sorted(regions, key=lambda r: (min(r)[0], min(r)[1]))

def transform_grid(input_grid):
    color_key = extract_color_key(input_grid)
    sorted_regions = get_sorted_regions(input_grid)
    
    new_grid = [row[:] for row in input_grid]  # Create a deep copy of the input grid
    
    # Assign new colors to regions
    for i, region in enumerate(sorted_regions):
        new_color = color_key[i % len(color_key)]
        for r, c in region:
            new_grid[r][c] = new_color
    
    return new_grid

def solve_1da012fc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Extracting the ordered list of colors from the first row containing gray (5) and non-black, non-gray colors.
    2. Identifying and sorting non-gray, non-black regions in the grid based on their top-left coordinate.
    3. Assigning new colors to the sorted regions cyclically using the extracted color key.
    4. Creating a new grid with the transformed colors while preserving the original structure.
    5. Returning the transformed grid.
    """
    transformed_values = transform_grid(input_grid.values)
    return ColoredGrid(values=transformed_values)
