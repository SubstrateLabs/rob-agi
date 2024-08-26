from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_cb227835(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by creating a zigzag path between two sky-colored squares.
    
    The function finds the two sky-colored (8) squares in the input grid,
    creates a zigzag path between them, and marks the path with green (3) squares.
    The path alternates between moving diagonally and straight (horizontally or vertically).
    The original sky squares remain unchanged.
    
    The zigzag path is created by alternating between diagonal moves and straight moves
    (either horizontal or vertical, depending on the relative positions of the start and end points).
    This creates a path that zig-zags towards the destination, filling the space between the two sky squares.
    
    Args:
    input_grid (ColoredGrid): The input grid containing two sky-colored squares.
    
    Returns:
    ColoredGrid: A new grid with the zigzag path marked in green.
    """
    def find_sky_squares(grid: List[List[int]]) -> List[Tuple[int, int]]:
        return [(c, r) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 8]
    
    def create_zigzag_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        current = start
        move_diagonal = True
        while current != end:
            x, y = current
            dx = 1 if end[0] > x else -1 if end[0] < x else 0
            dy = 1 if end[1] > y else -1 if end[1] < y else 0
            if move_diagonal:
                if dx != 0 and dy != 0:
                    current = (x + dx, y + dy)
                elif dx != 0:
                    current = (x + dx, y)
                else:
                    current = (x, y + dy)
            else:
                if dx != 0:
                    current = (x + dx, y)
                elif dy != 0:
                    current = (x, y + dy)
                else:
                    break  # We've reached the end
            path.append(current)
            move_diagonal = not move_diagonal
        return path
    
    sky_squares = find_sky_squares(input_grid.values)
    if len(sky_squares) != 2:
        return input_grid  # Return original grid if there aren't exactly two sky squares
    
    start, end = sky_squares
    path = create_zigzag_path(start, end)
    
    new_grid = input_grid.deep_copy()
    for x, y in path[1:-1]:  # Exclude start and end points
        new_grid.values[y][x] = 3  # Set to green
    
    return new_grid
