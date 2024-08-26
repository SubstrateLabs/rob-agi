from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_11e1fe23(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting the top three colored dots with a Z-shaped zigzag line.
    
    1. Identifies all colored dots in the grid.
    2. Selects the top three dots based on their vertical position.
    3. Creates a Z-shaped zigzag path connecting these dots:
       - Vertical line from top dot to midpoint
       - Horizontal line across midpoint
       - Vertical line from midpoint to bottom dot
    4. Calculates a new color for the middle segment of the zigzag.
    5. Draws the path on a copy of the input grid.
    
    Returns the new grid with the added zigzag path.
    """
    # Step 1: Scan the input grid
    colored_dots = [(r, c, input_grid.get_cell(r, c)) 
                    for r in range(input_grid.num_rows) 
                    for c in range(input_grid.num_cols) 
                    if input_grid.get_cell(r, c) != 0]
    
    if len(colored_dots) < 3:
        return input_grid  # Not enough dots to form a zigzag
    
    # Step 2: Sort dots vertically and select top 3
    sorted_dots = sorted(colored_dots, key=lambda x: x[0])[:3]
    
    # Step 3: Create zigzag path
    dot1, dot2, dot3 = sorted_dots
    midpoint_row = (dot1[0] + dot3[0]) // 2
    middle_color = (dot1[2] + dot2[2]) % 10
    
    path = [
        (dot1[0], dot1[1], dot1[2]),
        (midpoint_row, dot1[1], dot1[2]),
        (midpoint_row, dot1[1], middle_color),
        (midpoint_row, dot3[1], middle_color),
        (midpoint_row, dot3[1], dot3[2]),
        (dot3[0], dot3[1], dot3[2])
    ]
    
    # Step 4: Draw the path
    new_grid = input_grid.deep_copy()
    for i in range(len(path) - 1):
        draw_line(new_grid, path[i], path[i + 1], path[i][2])
    
    return new_grid

def draw_line(grid: ColoredGrid, start: Tuple[int, int, int], end: Tuple[int, int, int], color: int):
    """Draw a line between two points using Bresenham's line algorithm."""
    x0, y0 = start[1], start[0]
    x1, y1 = end[1], end[0]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if 0 <= y0 < grid.num_rows and 0 <= x0 < grid.num_cols:
            grid.set_cell(y0, x0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
