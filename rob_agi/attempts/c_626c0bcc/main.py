from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_626c0bcc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring sky-colored (8) regions with a specific pattern.
    
    The algorithm works as follows:
    1. Identify all connected sky-colored regions.
    2. Analyze each region to determine its type (large, thin, or small).
    3. Color each region based on its type:
       - Large regions: Place 2x2 blue (1) squares in corners and fill with a specific pattern.
       - Thin regions: Use a fixed pattern based on the region's shape.
       - Small regions: Use a fixed pattern.
    4. Resolve any remaining color conflicts.
    
    This approach ensures no adjacent cells (including diagonally) have the same non-black color,
    while maintaining the overall shape of the original sky-colored regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sky_regions = input_grid.find_connected_regions(8)
    
    for region in sky_regions:
        region_type, blue_locations = analyze_region(region)
        color_region(output_grid, region, region_type, blue_locations)
    
    resolve_all_conflicts(output_grid)
    return output_grid

def analyze_region(region: List[Tuple[int, int]]) -> Tuple[str, List[Tuple[int, int]]]:
    min_row, min_col = min(region)
    max_row, max_col = max(region)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    
    if width >= 3 and height >= 3:
        blue_locations = [(min_row, min_col), (min_row, max_col), (max_row, min_col), (max_row, max_col)]
        return "large", [loc for loc in blue_locations if loc in region]
    elif width <= 2 or height <= 2:
        return "thin", []
    else:
        return "small", []

def color_region(grid: ColoredGrid, region: List[Tuple[int, int]], region_type: str, blue_locations: List[Tuple[int, int]]):
    if region_type == "large":
        for r, c in blue_locations:
            place_shape(grid, region, (2, 2), 1, r, c)
        colors = [2, 3, 4]  # red, green, yellow
        color_index = 0
        for r, c in region:
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, colors[color_index])
                color_index = (color_index + 1) % 3
    elif region_type == "thin":
        color_thin_region(grid, region)
    else:  # small
        colors = [2, 3, 4, 2]
        for i, (r, c) in enumerate(region):
            grid.set_cell(r, c, colors[i % len(colors)])

def color_thin_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    min_row, min_col = min(region)
    max_row, max_col = max(region)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    
    if width <= 2:  # Vertical thin region
        colors = [2, 1, 3, 1]
        for i, (r, c) in enumerate(sorted(region)):
            grid.set_cell(r, c, colors[i % len(colors)])
    else:  # Horizontal thin region
        colors = [2, 4, 1, 1]
        for i, (r, c) in enumerate(sorted(region, key=lambda x: x[1])):
            grid.set_cell(r, c, colors[i % len(colors)])

def place_shape(grid: ColoredGrid, region: List[Tuple[int, int]], shape: Tuple[int, int], color: int, row: int, col: int):
    for r in range(row, row + shape[0]):
        for c in range(col, col + shape[1]):
            if (r, c) in region:
                grid.set_cell(r, c, color)

def resolve_all_conflicts(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                resolve_conflict(grid, r, c)

def resolve_conflict(grid: ColoredGrid, row: int, col: int):
    current_color = grid.get_cell(row, col)
    neighbors = get_neighbors(grid, row, col)
    neighbor_colors = set(grid.get_cell(r, c) for r, c in neighbors if grid.get_cell(r, c) != 0)
    
    if current_color in neighbor_colors:
        for new_color in [1, 2, 3, 4]:
            if new_color not in neighbor_colors:
                grid.set_cell(row, col, new_color)
                break

def get_neighbors(grid: ColoredGrid, row: int, col: int) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbors.append((r, c))
    return neighbors
