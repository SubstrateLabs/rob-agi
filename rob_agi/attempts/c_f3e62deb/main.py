from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_f3e62deb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by moving a 3x3 hollow square shape.
    
    The function identifies the 3x3 hollow square in the input grid and moves it
    according to the following priority order:
    1. If not on the right edge, move to the right edge.
    2. If on the right edge but not on the top edge, move to the top edge.
    3. If on the right and top edges but not on the left edge, move to the left edge.
    4. If on the right, top, and left edges but not on the bottom edge, move to the bottom edge.
    5. If on all edges, keep the current position.
    
    The shape maintains its vertical or horizontal position when moving to an edge.
    """
    def find_square(grid: ColoredGrid) -> Tuple[int, int, int]:
        for r in range(len(grid.values)):
            for c in range(len(grid.values[0])):
                if grid.values[r][c] != 0:
                    if r + 2 < len(grid.values) and c + 2 < len(grid.values[0]):
                        color = grid.values[r][c]
                        if all(grid.values[r+i][c+j] == color for i, j in [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]) and grid.values[r+1][c+1] == 0:
                            return r, c, color
        return -1, -1, -1

    top, left, color = find_square(input_grid)
    if top == -1:
        return input_grid  # No valid square found, return input grid unchanged

    right = left + 2
    bottom = top + 2

    # Determine new position
    if right < 9:  # Not on right edge
        new_left = 7
        new_top = top
    elif top > 0:  # On right edge but not on top
        new_left = left
        new_top = 0
    elif left > 0:  # On right and top edges but not on left
        new_left = 0
        new_top = top
    elif bottom < 9:  # On right, top, and left edges but not on bottom
        new_left = left
        new_top = 7
    else:  # On all edges
        new_left = left
        new_top = top

    # Create new grid with moved square
    new_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # Skip the center (keep it black)
            new_grid.values[new_top + i][new_left + j] = color

    return new_grid
