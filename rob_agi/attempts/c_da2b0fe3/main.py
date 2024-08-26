from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_da2b0fe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by adding a green line to the input grid.
    The line is placed either horizontally or vertically based on the orientation of the main shape in the grid.
    
    1. Find the bounding box of the main shape in the grid.
    2. Determine whether to add a horizontal or vertical green line based on the shape's orientation.
    3. Add the green line in the middle of the grid (5th row or column in a 10x10 grid).
    4. Return the modified grid with the added green line.
    """
    new_grid = input_grid.deep_copy()
    
    # Find the bounding box of the main shape
    top, left, bottom, right = find_bounding_box(new_grid.values)
    
    # Determine the orientation based on the shape's dimensions
    height = bottom - top + 1
    width = right - left + 1
    orientation = "horizontal" if height > width else "vertical"
    
    # Add the green line based on orientation
    if orientation == "horizontal":
        for c in range(len(new_grid.values[0])):
            new_grid.values[4][c] = 3
    else:
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
