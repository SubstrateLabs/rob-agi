from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions change to Red (2) if they are not part of a straight line (horizontal, vertical, or diagonal)
    2. Blue (1) regions that form straight lines remain Blue
    3. Red (2) regions remain unchanged
    4. All other colors remain unchanged

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Iterate through the grid to find connected regions of Blue (1) color.
    3. For each Blue region:
       a. Check if it forms a straight line (horizontal, vertical, or diagonal).
       b. If it doesn't, change all cells in the region to Red (2).
       c. If it does, leave it as Blue (1).
    4. Return the transformed grid.
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

    def is_straight_line(region: List[Tuple[int, int]]) -> bool:
        if len(region) < 2:
            return True
        
        r_coords = [r for r, _ in region]
        c_coords = [c for _, c in region]
        
        # Check if all points are on the same row, column, or diagonal
        return (len(set(r_coords)) == 1 or  # Horizontal line
                len(set(c_coords)) == 1 or  # Vertical line
                len(set(r - c for r, c in region)) == 1 or  # Diagonal (top-left to bottom-right)
                len(set(r + c for r, c in region)) == 1)  # Diagonal (top-right to bottom-left)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.values[r][c] == 1:  # Blue
                region = dfs(r, c)
                if not is_straight_line(region):
                    for cell_r, cell_c in region:
                        output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red

    return output_grid
