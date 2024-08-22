from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions that form a single straight line (horizontal, vertical, or diagonal) remain blue.
    2. Blue (1) regions that do not form a single straight line (including crosses, T-shapes, and L-shapes) change to red (2).
    3. All other colors remain unchanged.

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Find all connected blue regions using flood-fill algorithm.
    3. For each blue region:
       a. Check if the entire region forms a single straight line (horizontal, vertical, or diagonal).
       b. If it does, keep it Blue (1).
       c. If it doesn't (including crosses, T-shapes, and L-shapes), change all cells in the region to Red (2).
    4. Return the transformed grid.

    Implementation details:
    - The flood_fill function identifies connected blue regions efficiently using a set for visited cells.
    - The is_single_straight_line function correctly identifies all types of straight lines (horizontal, vertical, and diagonal).
    - The process_region function changes non-straight blue regions to red.
    - Edge cases for single-cell and two-cell regions are handled correctly.

    This implementation correctly identifies and transforms all blue regions,
    including complex shapes like L-shapes and T-shapes, while keeping true straight
    lines (horizontal, vertical, and diagonal) unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def flood_fill(r: int, c: int) -> Set[Tuple[int, int]]:
        region = set()
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == 1:  # Blue
                visited.add((curr_r, curr_c))
                region.add((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    def is_single_straight_line(region: Set[Tuple[int, int]]) -> bool:
        if len(region) <= 2:
            return True
        
        points = list(region)
        rows = set(p[0] for p in points)
        cols = set(p[1] for p in points)
        
        # Check if all points are in the same row or column
        if len(rows) == 1 or len(cols) == 1:
            return True
        
        # Check if it's a diagonal line
        if len(rows) == len(cols) == len(region):
            sorted_points = sorted(points)
            first, last = sorted_points[0], sorted_points[-1]
            dx, dy = last[0] - first[0], last[1] - first[1]
            if abs(dx) == abs(dy):
                return all((p[0] - first[0]) * dy == (p[1] - first[1]) * dx for p in sorted_points[1:-1])
        
        return False

    def process_region(region: Set[Tuple[int, int]]) -> None:
        if not is_single_straight_line(region):
            for cell_r, cell_c in region:
                output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.values[r][c] == 1:  # Blue
                region = flood_fill(r, c)
                process_region(region)

    return output_grid
