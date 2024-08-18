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

    Improvements:
    - The is_straight_line function now uses a more robust algorithm to check if all points lie on a single line.
    - The is_single_path function has been removed, as the straight line check implicitly ensures a single path.
    - Edge cases for single-cell and two-cell regions are handled separately for efficiency.
    - The flood-fill algorithm has been optimized to use a set for visited cells, improving performance.

    This implementation should correctly identify and transform all blue regions,
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    def is_straight_line(line: Set[Tuple[int, int]]) -> bool:
        if len(line) <= 2:
            return True
    
        points = list(line)
        p1, p2 = points[0], points[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        for point in points[1:-1]:
            cross_product = (point[0] - p1[0]) * dy - (point[1] - p1[1]) * dx
            if cross_product != 0:
                return False
        
        return True

    def process_region(region: Set[Tuple[int, int]]) -> None:
        if not is_straight_line(region):
            for cell_r, cell_c in region:
                output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.values[r][c] == 1:  # Blue
                region = flood_fill(r, c)
                process_region(region)

    return output_grid
