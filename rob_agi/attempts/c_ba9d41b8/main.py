from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ba9d41b8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a checkerboard pattern to non-black regions.
    The outer border of each region remains unchanged, while the inner part is filled
    with a checkerboard pattern using the original color and black (0).
    
    1. Identifies non-black regions in the grid.
    2. Processes each region by applying the checkerboard pattern, preserving the border.
    3. Returns the modified grid.

    The checkerboard pattern is applied based on the relative position of each cell
    within its region, ensuring consistent patterning for all regions regardless of their position.
    The top-left corner of each region's interior always keeps its original color.
    """
    if not input_grid.is_valid:
        raise ValueError("Invalid input grid")

    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_regions() -> List[Tuple[int, int, int, int, int]]:
        regions = []
        visited = set()

        def dfs(r: int, c: int, color: int) -> Tuple[int, int, int, int]:
            stack = [(r, c)]
            min_r, min_c, max_r, max_c = r, c, r, c
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and output_grid.get_cell(curr_r, curr_c) == color:
                    visited.add((curr_r, curr_c))
                    min_r, min_c = min(min_r, curr_r), min(min_c, curr_c)
                    max_r, max_c = max(max_r, curr_r), max(max_c, curr_c)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return min_r, min_c, max_r, max_c

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and output_grid.get_cell(r, c) != 0:
                    color = output_grid.get_cell(r, c)
                    top, left, bottom, right = dfs(r, c, color)
                    regions.append((color, top, left, bottom, right))

        return regions

    def process_region(color: int, top: int, left: int, bottom: int, right: int):
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if r == top or r == bottom or c == left or c == right:
                    continue  # Preserve the border
                if ((r - top) + (c - left)) % 2 == 1:
                    output_grid.set_cell(r, c, 0)

    regions = find_regions()
    for region in regions:
        process_region(*region)

    return output_grid
