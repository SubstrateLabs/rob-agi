from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by creating a diagonal path of 8 sky blue squares
    through the black region of the grid, avoiding obstacles and connecting opposite corners.

    1. Identify the starting corner (top-left or top-right) based on the black region
    2. Use a modified depth-first search to find a diagonal path of exactly 8 steps
    3. Prioritize diagonal moves and avoid gray obstacles
    4. Ensure the path ends at the opposite corner (bottom-right or bottom-left)
    5. Place 8 sky blue squares along the found path
    6. Return the modified grid or the original if no valid path is found
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def is_corner(r, c):
        return (r == 0 and c == 0) or (r == 0 and c == cols-1) or (r == rows-1 and c == 0) or (r == rows-1 and c == cols-1)

    def get_neighbors(r, c, direction):
        neighbors = []
        diag_dr, diag_dc = direction
        for dr, dc in [(diag_dr, diag_dc), (diag_dr, 0), (0, diag_dc)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == 0:
                neighbors.append((nr, nc))
        return neighbors

    def dfs(r, c, path, direction):
        if len(path) == 8:
            return [path] if is_corner(r, c) else []
        
        paths = []
        for nr, nc in get_neighbors(r, c, direction):
            if (nr, nc) not in path:
                new_path = path + [(nr, nc)]
                paths.extend(dfs(nr, nc, new_path, direction))
        return paths

    # Determine starting corner and direction
    if input_grid.get_cell(0, 0) == 0:
        start = (0, 0)
        direction = (1, 1)
    else:
        start = (0, cols-1)
        direction = (1, -1)

    best_path = dfs(*start, [start], direction)[0] if dfs(*start, [start], direction) else None

    if best_path:
        for r, c in best_path:
            output_grid.set_cell(r, c, 8)
        return output_grid
    else:
        return input_grid  # Unable to find a solution, return original grid
