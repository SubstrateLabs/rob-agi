from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions change to Red (2) if they are not a single straight line (horizontal, vertical, or diagonal)
    2. Blue (1) regions that form a single straight line remain Blue
    3. Red (2) regions remain unchanged
    4. All other colors remain unchanged

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Find all connected blue regions using depth-first search (DFS).
    3. For each blue region:
       a. Check if the entire region forms a single straight line.
       b. If it does, keep it Blue (1).
       c. If it doesn't, change all cells in the region to Red (2).
    4. Return the transformed grid.

    This implementation ensures that only blue regions that form a single
    straight line remain blue, while all other blue regions (including crosses
    and T-shapes) are changed to red.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == 1:  # Blue
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    def is_straight_line(line: List[Tuple[int, int]]) -> bool:
        if len(line) <= 2:
            return True
        
        r_coords, c_coords = zip(*line)
        
        is_horizontal = len(set(r_coords)) == 1
        is_vertical = len(set(c_coords)) == 1
        
        diffs = [r - c for r, c in line]
        sums = [r + c for r, c in line]
        is_diagonal_1 = len(set(diffs)) == 1
        is_diagonal_2 = len(set(sums)) == 1
        
        return is_horizontal or is_vertical or is_diagonal_1 or is_diagonal_2

    def process_region(region: List[Tuple[int, int]]) -> None:
        if len(region) <= 2 or is_straight_line(region):
            return  # Keep the region blue if it's a single straight line
        for cell_r, cell_c in region:
            output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.values[r][c] == 1:  # Blue
                region = dfs(r, c)
                process_region(region)

    return output_grid
