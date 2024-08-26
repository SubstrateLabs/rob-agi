from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_0d87d2a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue dots vertically and filling the area to the left.
    
    1. Finds all blue (1) dots in the grid.
    2. Creates a vertical blue path connecting all blue dots.
    3. Fills all cells to the left of the blue path with blue.
    4. Preserves original blue dots and red blocks.
    5. Handles edge cases like single blue dot or no blue dots.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find blue dots
    blue_dots = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    if not blue_dots:
        return output_grid
    
    # Step 2: Create vertical blue path
    blue_dots.sort()
    for i in range(len(blue_dots)):
        r1, c1 = blue_dots[i]
        r2, c2 = blue_dots[(i + 1) % len(blue_dots)]
        for r in range(min(r1, r2), max(r1, r2) + 1):
            output_grid.values[r][c1] = 1
    
    # Add final vertical segment to the top
    for r in range(blue_dots[-1][0]):
        output_grid.values[r][blue_dots[-1][1]] = 1
    
    # Step 3: Fill from the left
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [1, 2]:
                break
            output_grid.values[r][c] = 1
    
    # Step 4: Preserve original elements
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                output_grid.values[r][c] = 1
            elif input_grid.values[r][c] == 2 and output_grid.values[r][c] != 1:
                output_grid.values[r][c] = 2
    
    return output_grid
