from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_da2b0fe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by adding a green line to the input grid.
    The line is always placed vertically in the middle column (5th column in a 10x10 grid).
    The green line extends from the top to the bottom of the grid, regardless of the content.
    
    1. Create a deep copy of the input grid.
    2. Add a vertical green line in the middle column (index 4 for a 0-indexed grid).
    3. Return the modified grid with the added green line.
    """
    new_grid = input_grid.deep_copy()
    
    # Add the vertical green line in the middle column (5th column, index 4)
    for r in range(len(new_grid.values)):
        new_grid.values[r][4] = 3
    
    return new_grid

def find_bounding_box(grid: List[List[int]]) -> Tuple[int, int, int, int]:
    rows, cols = len(grid), len(grid[0])
    top, left = rows, cols
    bottom, right = -1, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                top = min(top, r)
                left = min(left, c)
                bottom = max(bottom, r)
                right = max(right, c)
    
    return top, left, bottom, right
