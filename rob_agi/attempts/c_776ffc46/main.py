from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions that form a single straight line (horizontal, vertical, or diagonal) remain blue.
    2. Blue (1) regions that do not form a single straight line (including crosses, T-shapes, and L-shapes) change to red (2).
    3. All other colors remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def is_straight_line(region: Set[Tuple[int, int]]) -> bool:
        if len(region) <= 2:
            return True
        
        points = list(region)
        first, last = min(points), max(points)
        dx, dy = last[0] - first[0], last[1] - first[1]
        
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):  # Vertical, horizontal, or diagonal
            return all((p[0] - first[0]) * dy == (p[1] - first[1]) * dx for p in points)
        
        return False

    def flood_fill(r: int, c: int) -> Set[Tuple[int, int]]:
        region = set()
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and output_grid.get_cell(curr_r, curr_c) == 1:  # Blue
                visited.add((curr_r, curr_c))
                region.add((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only orthogonal neighbors
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.get_cell(r, c) == 1:  # Blue
                region = flood_fill(r, c)
                if not is_straight_line(region):
                    for cell_r, cell_c in region:
                        output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red (2)

    return output_grid
