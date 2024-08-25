from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_45737921(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 45737921 challenge by swapping colors within each connected region of the grid.
    
    The solution works as follows:
    1. Create a deep copy of the input grid.
    2. Find all non-black connected regions in the grid.
    3. For each region with exactly two colors, swap those colors.
    4. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_all_regions(output_grid)
    for region in regions:
        process_region(output_grid, region)
    return output_grid

def find_all_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols) or grid.get_cell(r, c) != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc, color))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                regions.append(dfs(r, c, grid.get_cell(r, c)))

    return regions

def process_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    colors = get_unique_colors(grid, region)
    if len(colors) == 2:
        swap_colors(grid, region, colors)

def get_unique_colors(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Set[int]:
    return set(grid.get_cell(r, c) for r, c in region)

def swap_colors(grid: ColoredGrid, region: List[Tuple[int, int]], colors: Set[int]):
    color_list = list(colors)
    swap_map = {color_list[0]: color_list[1], color_list[1]: color_list[0]}
    for r, c in region:
        current_color = grid.get_cell(r, c)
        new_color = swap_map[current_color]
        grid.set_cell(r, c, new_color)
