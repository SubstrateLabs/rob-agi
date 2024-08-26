from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_cb227835(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by creating an optimal path between two sky-colored squares.
    
    The function finds the two sky-colored (8) squares in the input grid,
    creates a path between them, and marks the path with green (3) squares.
    The path is chosen based on the relative positions of the start and end points:
    - Straight line for aligned squares
    - Rectangular path for squares forming a rectangle
    - Symmetrical zigzag for perfect diagonals
    - Adaptive rectangular path for other cases
    
    The original sky squares remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid containing two sky-colored squares.
    
    Returns:
    ColoredGrid: A new grid with the optimal path marked in green.
    """
    def find_sky_squares(grid: List[List[int]]) -> List[Tuple[int, int]]:
        return [(c, r) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 8]
    
    def create_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        x, y = start
        ex, ey = end
        dx = ex - x
        dy = ey - y
        
        if dx == 0 or dy == 0:  # Straight line
            step_x = 1 if dx > 0 else -1 if dx < 0 else 0
            step_y = 1 if dy > 0 else -1 if dy < 0 else 0
            while (x, y) != end:
                x += step_x
                y += step_y
                path.append((x, y))
        elif abs(dx) == abs(dy):  # Perfect diagonal (symmetrical zigzag)
            step_x = 1 if dx > 0 else -1
            step_y = 1 if dy > 0 else -1
            while (x, y) != end:
                x += step_x
                path.append((x, y))
                if (x, y) != end:
                    y += step_y
                    path.append((x, y))
        else:  # Adaptive rectangular path
            mid_x = (x + ex) // 2
            while x != mid_x:
                x += 1 if dx > 0 else -1
                path.append((x, y))
            while y != ey:
                y += 1 if dy > 0 else -1
                path.append((x, y))
            while x != ex:
                x += 1 if dx > 0 else -1
                path.append((x, y))
        
        return path
    
    sky_squares = find_sky_squares(input_grid.values)
    if len(sky_squares) != 2:
        return input_grid  # Return original grid if there aren't exactly two sky squares
    
    start, end = sky_squares
    path = create_path(start, end)
    
    new_grid = input_grid.deep_copy()
    for x, y in path:
        if new_grid.values[y][x] != 8:  # Don't overwrite sky squares
            new_grid.values[y][x] = 3  # Set to green
    
    return new_grid
