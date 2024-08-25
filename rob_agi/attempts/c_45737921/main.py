from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_45737921(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 45737921 challenge by reversing the order of colors within each connected region of the grid that contains exactly two colors.
    
    The solution works as follows:
    1. Create a deep copy of the input grid.
    2. Find all non-black connected regions in the grid.
    3. For each region with exactly two colors:
       - Separate the cells of each color into two lists.
       - Sort each list based on row-major order.
       - Reverse both sorted lists.
       - Reassign colors to the cells based on the reversed lists.
    4. Return the modified grid.
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
    colors = list(set(grid.get_cell(r, c) for r, c in region))
    color_lists = [[], []]
    for r, c in region:
        color_index = colors.index(grid.get_cell(r, c))
        color_lists[color_index].append((r, c))
    
    for color_list in color_lists:
        color_list.sort(key=lambda x: (x[0], x[1]))  # Sort by row, then column
        color_list.reverse()
    
    for (r1, c1), (r2, c2) in zip(color_lists[0], color_lists[1]):
        color1 = grid.get_cell(r1, c1)
        color2 = grid.get_cell(r2, c2)
        grid.set_cell(r1, c1, color2)
        grid.set_cell(r2, c2, color1)
