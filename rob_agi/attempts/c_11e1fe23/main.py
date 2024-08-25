from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_11e1fe23(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a diamond shape between the two horizontally closest colored dots.
    
    1. Identifies all colored dots in the grid.
    2. Finds the two horizontally closest colored dots.
    3. Determines the diamond shape between these dots.
    4. Colors the diamond based on the colors of the closest dots.
    5. Draws the diamond on a copy of the input grid.
    
    Returns the new grid with the added diamond shape.
    """
    # Step 1: Scan the input grid
    colored_dots = [(r, c, input_grid.get_cell(r, c)) 
                    for r in range(input_grid.num_rows) 
                    for c in range(input_grid.num_cols) 
                    if input_grid.get_cell(r, c) != 0]
    
    if len(colored_dots) < 2:
        return input_grid  # Not enough dots to form a diamond
    
    # Step 2: Find the two horizontally closest colored dots
    closest_pair = min([(a, b) for a in colored_dots for b in colored_dots if a != b],
                       key=lambda pair: abs(pair[0][1] - pair[1][1]))
    
    # Step 3: Determine the diamond shape
    left_dot, right_dot = sorted(closest_pair, key=lambda dot: dot[1])
    mid_row = (left_dot[0] + right_dot[0]) // 2
    mid_col = (left_dot[1] + right_dot[1]) // 2
    height = max(1, abs(left_dot[0] - right_dot[0]) // 2)
    
    # Step 4: Color the diamond
    top_color = left_dot[2]
    bottom_color = right_dot[2]
    left_color = 5  # Gray
    right_color = (left_dot[2] + right_dot[2]) % 10
    
    # Step 5: Draw the diamond
    new_grid = input_grid.deep_copy()
    diamond_points = [
        (mid_row - height, mid_col, top_color),
        (mid_row, left_dot[1], left_color),
        (mid_row, right_dot[1], right_color),
        (mid_row + height, mid_col, bottom_color)
    ]
    
    for point in diamond_points:
        if 0 <= point[0] < new_grid.num_rows and 0 <= point[1] < new_grid.num_cols:
            new_grid.set_cell(point[0], point[1], point[2])
    
    # Draw lines connecting the diamond points
    draw_line(new_grid, diamond_points[0], diamond_points[1], top_color)
    draw_line(new_grid, diamond_points[0], diamond_points[2], right_color)
    draw_line(new_grid, diamond_points[3], diamond_points[1], left_color)
    draw_line(new_grid, diamond_points[3], diamond_points[2], bottom_color)
    
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
