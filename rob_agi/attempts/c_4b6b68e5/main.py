from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_4b6b68e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling enclosed regions with sky blue (8) or magenta (6).
    
    The solution follows these steps:
    1. Identify all enclosed regions in the grid for each non-zero color.
    2. For each enclosed region:
       a. If it contains or is adjacent to sky blue (8), fill the entire region with 8.
       b. If it contains or is adjacent to magenta (6), fill the entire region with 6.
       c. If neither 8 nor 6 is present, leave the region unchanged.
    3. Return the modified grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid after applying the filling rules.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_enclosed(region: Set[Tuple[int, int]], color: int) -> bool:
        for r, c in region:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.get_cell(nr, nc) not in [0, color]:
                        return False
                else:
                    return False
        return True

    def flood_fill(r: int, c: int, old_color: int, new_color: int):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            output_grid.get_cell(r, c) != old_color):
            return
        output_grid.set_cell(r, c, new_color)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            flood_fill(r + dr, c + dc, old_color, new_color)

    def region_contains_or_adjacent(region: Set[Tuple[int, int]], target_color: int) -> bool:
        for r, c in region:
            if input_grid.get_cell(r, c) == target_color:
                return True
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.get_cell(nr, nc) == target_color:
                        return True
        return False

    colors = set(color for row in input_grid.values for color in row if color != 0)
    
    for color in colors:
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            if is_enclosed(region, color):
                if region_contains_or_adjacent(region, 8):
                    r, c = next(iter(region))
                    flood_fill(r, c, color, 8)
                elif region_contains_or_adjacent(region, 6):
                    r, c = next(iter(region))
                    flood_fill(r, c, color, 6)
    
    return output_grid
