from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function finds all red dots (value 2) in the input grid, determines their bounding box,
    draws a green (value 3) rectangle around this bounding box, and then connects internal red dots
    to the nearest edge of the bounding box. This creates a closed shape that encompasses all red dots
    while minimizing internal lines.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming a closed shape.
    """
    # Create a deep copy of the input grid
    result_grid = input_grid.deep_copy()
    
    # Find all red dots
    red_dots = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.values[r][c] == 2]
    
    if not red_dots:
        return result_grid  # No red dots to connect
    
    # Determine the bounding box
    min_r = min(r for r, _ in red_dots)
    max_r = max(r for r, _ in red_dots)
    min_c = min(c for _, c in red_dots)
    max_c = max(c for _, c in red_dots)
    
    # Draw the outer rectangle (bounding box)
    draw_line(result_grid, (min_r, min_c), (min_r, max_c))  # Top
    draw_line(result_grid, (max_r, min_c), (max_r, max_c))  # Bottom
    draw_line(result_grid, (min_r, min_c), (max_r, min_c))  # Left
    draw_line(result_grid, (min_r, max_c), (max_r, max_c))  # Right
    
    # Connect internal dots to the bounding box
    for r, c in red_dots:
        if r > min_r and r < max_r and c > min_c and c < max_c:
            # Determine the nearest edge
            dist_top = r - min_r
            dist_bottom = max_r - r
            dist_left = c - min_c
            dist_right = max_c - c
            
            if min(dist_top, dist_bottom) <= min(dist_left, dist_right):
                # Connect vertically
                if dist_top <= dist_bottom:
                    draw_line(result_grid, (r, c), (min_r, c))
                else:
                    draw_line(result_grid, (r, c), (max_r, c))
            else:
                # Connect horizontally
                if dist_left <= dist_right:
                    draw_line(result_grid, (r, c), (r, min_c))
                else:
                    draw_line(result_grid, (r, c), (r, max_c))
    
    return result_grid

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Draw a line from start to end on the grid, avoiding overwriting red dots."""
    r1, c1 = start
    r2, c2 = end
    
    # Ensure r1 <= r2 and c1 <= c2
    if r1 > r2:
        r1, r2 = r2, r1
    if c1 > c2:
        c1, c2 = c2, c1
    
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if grid.values[r][c] == 0:  # Only fill black cells
                grid.values[r][c] = 3  # Green
