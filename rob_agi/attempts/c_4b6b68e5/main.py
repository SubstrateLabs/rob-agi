from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_4b6b68e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling enclosed regions with the highest-valued color
    present within the region or adjacent to it, excluding black (0).
    
    The solution follows these steps:
    1. Identify all enclosed regions in the grid for each non-zero color.
    2. For each enclosed region:
       a. Determine the highest-valued color within the region or adjacent to it (excluding black).
       b. Fill the entire region with this highest-valued color.
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
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if (0 <= r < rows and 0 <= c < cols and
                output_grid.get_cell(r, c) == old_color):
                output_grid.set_cell(r, c, new_color)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))

    def get_highest_color(region: Set[Tuple[int, int]], color: int) -> int:
        highest_color = color
        for r, c in region:
            highest_color = max(highest_color, input_grid.get_cell(r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_color = input_grid.get_cell(nr, nc)
                    if neighbor_color != 0:  # Exclude black
                        highest_color = max(highest_color, neighbor_color)
        return highest_color

    colors = set(color for row in input_grid.values for color in row if color != 0)
    
    for color in sorted(colors, reverse=True):
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            if is_enclosed(region, color):
                highest_color = get_highest_color(region, color)
                r, c = next(iter(region))
                flood_fill(r, c, color, highest_color)
    
    return output_grid
