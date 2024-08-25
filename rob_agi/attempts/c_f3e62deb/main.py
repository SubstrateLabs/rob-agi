from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_f3e62deb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by moving a 3x3 hollow square shape.
    
    The function identifies the 3x3 hollow square in the input grid and moves it
    according to the following rules:
    1. If not on the right edge, move to the right edge.
    2. If on the right edge but not on the top edge, move to the top edge.
    3. If on both right and top edges, move to the bottom edge.
    
    The shape maintains its vertical or horizontal position when moving to an edge.
    """
    def find_first_nonzero(grid: ColoredGrid) -> Tuple[int, int]:
        for r, row in enumerate(grid.values):
            for c, val in enumerate(row):
                if val != 0:
                    return r, c
        return -1, -1

    def is_valid_square(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
        if r + 2 >= len(grid.values) or c + 2 >= len(grid.values[0]):
            return False
        pattern = [
            [color, color, color],
            [color, 0, color],
            [color, color, color]
        ]
        return all(grid.values[r+i][c+j] == pattern[i][j] for i in range(3) for j in range(3))

    def get_square_center(r: int, c: int) -> Tuple[int, int]:
        return r + 1, c + 1

    # Find the square
    top, left = find_first_nonzero(input_grid)
    if top == -1 or not is_valid_square(input_grid, top, left, input_grid.values[top][left]):
        return input_grid  # No valid square found, return input grid unchanged

    color = input_grid.values[top][left]
    center_y, center_x = get_square_center(top, left)

    # Determine new position
    if center_x < 7:  # Not on right edge
        new_left = 7
        new_top = top
    elif center_y > 1:  # On right edge but not on top
        new_left = left
        new_top = 0
    else:  # On right and top edges
        new_left = left
        new_top = 7

    # Create new grid with moved square
    new_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(3):
        for j in range(3):
            new_grid.values[new_top + i][new_left + j] = input_grid.values[top + i][left + j]

    return new_grid
