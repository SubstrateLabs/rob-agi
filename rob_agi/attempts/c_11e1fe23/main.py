from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_11e1fe23(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting the three colored dots with a zigzag line.
    
    1. Identifies all colored dots in the grid.
    2. Selects the top, middle, and bottom dots based on their vertical position.
    3. Creates a zigzag path connecting these dots:
       - Vertical line from top dot to an adjusted midpoint
       - Horizontal line across the adjusted midpoint
       - Vertical line from the adjusted midpoint to the bottom dot
    4. Calculates a new color for the horizontal segment of the zigzag.
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
    
    # Step 2: Sort dots vertically and select top, middle, and bottom
    sorted_dots = sorted(colored_dots, key=lambda x: x[0])
    top_dot, middle_dot, bottom_dot = sorted_dots[0], sorted_dots[1], sorted_dots[-1]
    
    # Step 3: Determine the zigzag path
    midpoint_row = (top_dot[0] + bottom_dot[0]) // 2
    if middle_dot[0] < midpoint_row:
        midpoint_row = (top_dot[0] + middle_dot[0]) // 2
    elif middle_dot[0] > midpoint_row:
        midpoint_row = (middle_dot[0] + bottom_dot[0]) // 2
    
    # Step 4: Calculate the new color for the horizontal segment
    horizontal_color = ((top_dot[2] + middle_dot[2]) % 10) + 5
    if horizontal_color > 9:
        horizontal_color = horizontal_color % 10
    
    # Step 5: Create the path
    path = [
        (top_dot[0], top_dot[1], top_dot[2]),
        (midpoint_row, top_dot[1], top_dot[2]),
        (midpoint_row, middle_dot[1], horizontal_color),
        (midpoint_row, bottom_dot[1], horizontal_color),
        (bottom_dot[0], bottom_dot[1], bottom_dot[2])
    ]
    
    # Step 6: Draw the path
    new_grid = input_grid.deep_copy()
    for i in range(len(path) - 1):
        draw_line(new_grid, path[i], path[i + 1])
    
    return new_grid

def draw_line(grid: ColoredGrid, start: Tuple[int, int, int], end: Tuple[int, int, int]):
    """Draw a line between two points using Bresenham's line algorithm."""
    x0, y0, color = start[1], start[0], start[2]
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
        
        # Update color for horizontal segment
        if start[0] == end[0]:  # Horizontal line
            progress = abs(x0 - start[1]) / abs(end[1] - start[1])
            color = round(start[2] * (1 - progress) + end[2] * progress)
