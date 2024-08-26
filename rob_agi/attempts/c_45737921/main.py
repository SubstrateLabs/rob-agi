from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_45737921(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 45737921 challenge by swapping colors within each connected region of the grid that contains exactly two colors.
    
    The solution works as follows:
    1. Create a deep copy of the input grid.
    2. Find all non-black connected regions in the grid.
    3. For each region with exactly two colors:
       - Identify the two colors present in the region.
       - Group cells by their color.
       - Sort cells in each color group based on row-major order.
       - Swap the colors of the sorted cells, maintaining their positions.
    4. Return the modified grid.

    This approach preserves the shape and structure of each region while reversing the color pattern
    for regions with exactly two colors. Regions with one color or more than two colors remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    regions = find_all_regions(output_grid)
    for region in regions:
        if has_two_colors(output_grid, region):
            reverse_colors_in_region(output_grid, region)
    return output_grid

def find_all_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols) or grid.get_cell(r, c) == 0:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                regions.append(dfs(r, c))

    return regions

def has_two_colors(grid: ColoredGrid, region: List[Tuple[int, int]]) -> bool:
    colors = set(grid.get_cell(r, c) for r, c in region)
    return len(colors) == 2

def reverse_colors_in_region(grid: ColoredGrid, region: List[Tuple[int, int]]) -> None:
    colors = sorted(set(grid.get_cell(r, c) for r, c in region))
    if len(colors) != 2:
        return  # Only process regions with exactly two colors

    color_to_cells = {color: [] for color in colors}
    for r, c in region:
        color = grid.get_cell(r, c)
        color_to_cells[color].append((r, c))

    for color in colors:
        color_to_cells[color].sort(key=lambda x: (x[0], x[1]))  # Sort by row, then column

    new_color_assignments = []
    for color, cells in color_to_cells.items():
        other_color = colors[1] if color == colors[0] else colors[0]
        new_color_assignments.extend([(cell, other_color) for cell in reversed(cells)])

    for (r, c), new_color in new_color_assignments:
        grid.set_cell(r, c, new_color)
