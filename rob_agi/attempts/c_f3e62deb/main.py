from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_f3e62deb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by moving a 3x3 hollow square shape.
    
    The function identifies the 3x3 hollow square in the input grid and moves it
    to the nearest available edge in the following priority order:
    1. Right edge
    2. Top edge
    3. Left edge
    4. Bottom edge
    
    The shape maintains its vertical or horizontal position when moving to an edge.
    If the shape is already at all edges, it remains in its current position.
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

    # Calculate distances to edges
    dist_right = 9 - (left + 2)
    dist_top = top
    dist_left = left
    dist_bottom = 9 - (top + 2)

    # Determine target edge
    edges = [(dist_right, 'right'), (dist_top, 'top'), (dist_left, 'left'), (dist_bottom, 'bottom')]
    edges.sort(key=lambda x: x[0])  # Sort by distance
    target_edge = next((edge for dist, edge in edges if dist > 0), 'stay')

    # Calculate new position
    if target_edge == 'right':
        new_left, new_top = 7, top
    elif target_edge == 'top':
        new_left, new_top = left, 0
    elif target_edge == 'left':
        new_left, new_top = 0, top
    elif target_edge == 'bottom':
        new_left, new_top = left, 7
    else:  # 'stay'
        new_left, new_top = left, top

    # Create new grid with moved square
    new_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # Skip the center (keep it black)
            new_grid.values[new_top + i][new_left + j] = color

    return new_grid
