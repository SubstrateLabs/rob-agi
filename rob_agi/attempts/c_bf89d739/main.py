from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function finds all red dots (value 2) in the input grid and connects them
    using green lines (value 3) to form a compact shape. It creates horizontal
    and vertical connections between red dots, prioritizing the creation of a
    single connected component that encompasses all red dots.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming a compact shape.
    """
    result_grid = input_grid.deep_copy()
    red_dots = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.values[r][c] == 2]
    
    if not red_dots:
        return result_grid

    # Sort red dots by row, then by column
    red_dots.sort()

    # Connect dots horizontally
    for i in range(len(red_dots) - 1):
        r1, c1 = red_dots[i]
        r2, c2 = red_dots[i + 1]
        if r1 == r2:
            draw_line(result_grid, (r1, c1), (r2, c2))

    # Connect dots vertically
    red_dots.sort(key=lambda x: (x[1], x[0]))  # Sort by column, then by row
    for i in range(len(red_dots) - 1):
        r1, c1 = red_dots[i]
        r2, c2 = red_dots[i + 1]
        if c1 == c2:
            draw_line(result_grid, (r1, c1), (r2, c2))

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
