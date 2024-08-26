from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function finds all red dots (value 2) in the input grid, sorts them from top to bottom,
    and then connects them with green lines (value 3) to form a closed shape. The lines are drawn
    vertically and horizontally, prioritizing vertical connections when possible.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines.
    """
    # Create a deep copy of the input grid
    result_grid = input_grid.deep_copy()
    
    # Find all red dots
    red_dots = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.values[r][c] == 2]
    
    # Sort red dots by row (y-coordinate) and then by column (x-coordinate)
    red_dots.sort()
    
    # Connect the dots
    for i in range(len(red_dots)):
        current_dot = red_dots[i]
        next_dot = red_dots[(i + 1) % len(red_dots)]  # Wrap around to the first dot
        
        # Draw line from current dot to next dot
        draw_line(result_grid, current_dot, next_dot)
    
    return result_grid

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Draw a line from start to end on the grid."""
    x1, y1 = start
    x2, y2 = end
    
    # Draw vertical line
    for y in range(min(y1, y2), max(y1, y2) + 1):
        if grid.values[y][x1] == 0:
            grid.values[y][x1] = 3
    
    # Draw horizontal line
    for x in range(min(x1, x2), max(x1, x2) + 1):
        if grid.values[y2][x] == 0:
            grid.values[y2][x] = 3
