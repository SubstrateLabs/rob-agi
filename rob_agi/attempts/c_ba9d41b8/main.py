from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ba9d41b8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a checkerboard pattern to non-black regions.
    The outer border of each region remains unchanged, as well as the bottom-right cell of each region.
    The inner part is filled with a checkerboard pattern using the original color and black (0).
    
    The checkerboard pattern is applied based on the relative position within each region,
    ensuring consistent patterning for all regions regardless of their position in the grid.
    Interior cells where the sum of (row - top) and (column - left) is odd are set to black (0).
    """
    if not input_grid.is_valid:
        raise ValueError("Invalid input grid")

    output_grid = input_grid.deep_copy()
    regions = find_regions(output_grid)
    
    for color, region in regions:
        process_region(output_grid, color, region)
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]]]]:
    regions = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                region = []
                color = grid.get_cell(r, c)
                queue = [(r, c)]
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    if (curr_r, curr_c) in visited:
                        continue
                    visited.add((curr_r, curr_c))
                    region.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_r, new_c = curr_r + dr, curr_c + dc
                        if 0 <= new_r < rows and 0 <= new_c < cols and grid.get_cell(new_r, new_c) == color:
                            queue.append((new_r, new_c))
                regions.append((color, region))
    return regions

def process_region(grid: ColoredGrid, color: int, region: List[Tuple[int, int]]):
    top = min(r for r, _ in region)
    bottom = max(r for r, _ in region)
    left = min(c for _, c in region)
    right = max(c for _, c in region)
    
    for r, c in region:
        if r == top or r == bottom or c == left or c == right:
            continue  # Leave border cells unchanged
        if r == bottom and c == right:
            continue  # Leave bottom-right cell unchanged
        if (r - top + c - left) % 2 == 1:
            grid.set_cell(r, c, 0)  # Set to black
