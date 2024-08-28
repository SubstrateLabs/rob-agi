from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_f3e62deb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by moving a 3x3 hollow square shape.
    
    The function identifies the 3x3 hollow square in the input grid and moves it
    to the right edge if possible. If the square is already at the right edge,
    it moves to the bottom edge while maintaining its horizontal position.
    If it's already at the bottom edge, it moves to the top edge.
    """
    def find_hollow_square(grid: ColoredGrid) -> Tuple[int, int, int]:
        for r in range(8):
            for c in range(8):
                color = grid.values[r][c]
                if color != 0 and grid.extract_subgrid(r, c, 3, 3).values == [
                    [color, color, color],
                    [color, 0, color],
                    [color, color, color]
                ]:
                    return r, c, color
        return -1, -1, -1

    top, left, color = find_hollow_square(input_grid)
    if top == -1:
        return input_grid  # No valid square found, return input grid unchanged

    # Determine new position
    if left < 7:
        new_left, new_top = 7, top  # Move to right edge
    elif top < 7:
        new_left, new_top = left, 7  # Move to bottom edge
    else:
        new_left, new_top = left, 0  # Move to top edge

    # Create new grid with moved square
    new_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # Skip the center (keep it black)
            new_grid.values[new_top + i][new_left + j] = color

    return new_grid
