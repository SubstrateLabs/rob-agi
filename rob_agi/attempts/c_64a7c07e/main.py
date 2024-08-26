from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting non-black cells horizontally towards the center.
    
    The function calculates a uniform shift for all non-black cells based on the grid width.
    It then creates a new grid where each non-black cell is moved to the right by the calculated shift amount.
    The vertical positions and internal structure of all shapes are preserved.
    If a cell would be shifted out of bounds, it is placed at the rightmost edge of the grid.
    """
    height, width = input_grid.get_dimensions()
    shift = (width - 1) // 2
    
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    for r in range(height):
        for c in range(width):
            if input_grid.values[r][c] != 0:
                new_c = min(c + shift, width - 1)
                new_grid.values[r][new_c] = input_grid.values[r][c]
    
    return new_grid
