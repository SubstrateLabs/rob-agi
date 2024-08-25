from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_45737921(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 45737921 challenge by reorganizing colors within each connected region of the grid.
    
    The solution works as follows:
    1. Create a deep copy of the input grid.
    2. Find all non-black connected regions in the grid.
    3. For each region with exactly two colors:
       a. Determine the fill color (more frequent or lower-numbered if tied).
       b. Reorganize colors to create larger, more contiguous sub-regions:
          - Keep the fill color cells unchanged.
          - Change non-fill color cells to fill color, except for edge cells and
            those with diagonal neighbors of the non-fill color.
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

def process_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    colors = get_unique_colors(grid, region)
    if len(colors) == 2:
        reorganize_colors(grid, region, colors)

def get_unique_colors(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Set[int]:
    return set(grid.get_cell(r, c) for r, c in region)

def reorganize_colors(grid: ColoredGrid, region: List[Tuple[int, int]], colors: Set[int]):
    color_counts = count_colors(grid, region)
    fill_color = max(color_counts, key=color_counts.get)
    non_fill_color = (colors - {fill_color}).pop()

    for r, c in region:
        if grid.get_cell(r, c) == non_fill_color:
            if not is_edge_cell(grid, r, c, region) and not has_diagonal_neighbor(grid, r, c, non_fill_color):
                grid.set_cell(r, c, fill_color)

def count_colors(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Dict[int, int]:
    return {color: sum(1 for r, c in region if grid.get_cell(r, c) == color) for color in get_unique_colors(grid, region)}

def is_edge_cell(grid: ColoredGrid, row: int, col: int, region: List[Tuple[int, int]]) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = row + dr, col + dc
        if (nr, nc) not in region or not (0 <= nr < rows and 0 <= nc < cols) or grid.get_cell(nr, nc) == 0:
            return True
    return False

def has_diagonal_neighbor(grid: ColoredGrid, row: int, col: int, color: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == color:
            return True
    return False
