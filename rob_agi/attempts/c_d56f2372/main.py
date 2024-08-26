from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d56f2372(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by extracting the most prominent shape from the input grid.
    
    1. Identifies all non-zero color regions in the input grid.
    2. Calculates a prominence score for each region based on size, position, and shape complexity.
    3. Selects the region with the highest prominence score.
    4. Creates a new grid containing only the selected shape, preserving its relative position.
    
    Args:
        input_grid (ColoredGrid): The input grid to process.
    
    Returns:
        ColoredGrid: A new grid containing only the extracted shape.
    """
    regions = find_all_regions(input_grid)
    if not regions:
        return ColoredGrid(values=[[0]])
    
    most_prominent = max(regions, key=lambda r: calculate_prominence(r, input_grid))
    return extract_shape(most_prominent, input_grid)

def find_all_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    visited = set()
    regions = []
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = []
                dfs(grid, r, c, grid.values[r][c], visited, region)
                regions.append(region)
    return regions

def dfs(grid: ColoredGrid, r: int, c: int, color: int, visited: set, region: list):
    if (r, c) in visited or r < 0 or r >= grid.num_rows or c < 0 or c >= grid.num_cols or grid.values[r][c] != color:
        return
    visited.add((r, c))
    region.append((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, visited, region)

def calculate_prominence(region: List[Tuple[int, int]], grid: ColoredGrid) -> float:
    size_factor = len(region)
    
    center_r = sum(r for r, _ in region) / len(region)
    center_c = sum(c for _, c in region) / len(region)
    grid_center_r, grid_center_c = grid.num_rows / 2, grid.num_cols / 2
    position_factor = 1 / (1 + ((center_r - grid_center_r)**2 + (center_c - grid_center_c)**2)**0.5)
    
    perimeter = sum(1 for r, c in region if any((r+dr, c+dc) not in region for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]))
    shape_factor = perimeter / size_factor
    
    color_factor = grid.values[region[0][0]][region[0][1]] / 9  # Normalize color value
    
    return (size_factor * 0.4) + (position_factor * 0.3) + (shape_factor * 0.2) + (color_factor * 0.1)

def extract_shape(region: List[Tuple[int, int]], grid: ColoredGrid) -> ColoredGrid:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    color = grid.values[region[0][0]][region[0][1]]
    for r, c in region:
        new_grid.values[r - min_r][c - min_c] = color
    
    return new_grid
