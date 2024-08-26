from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function finds all red dots (value 2) in the input grid and connects them
    using green lines (value 3) to form a single connected path. It prioritizes
    vertical connections, then horizontal connections. The algorithm ensures that
    all red dots are connected in a way that forms a compact shape.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming a single path.
    """
    result_grid = input_grid.deep_copy()
    red_dots = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.values[r][c] == 2]
    
    if not red_dots:
        return result_grid

    # Sort red dots by row, then by column
    red_dots.sort(key=lambda coord: (coord[0], coord[1]))

    # Connect dots
    for i in range(len(red_dots) - 1):
        start = red_dots[i]
        end = red_dots[i + 1]
        connect_dots(result_grid, start, end)

    return result_grid

def connect_dots(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Connect two dots with green lines, prioritizing vertical connections."""
    y1, x1 = start
    y2, x2 = end

    # Draw vertical line
    draw_line(grid, (y1, x1), (y2, x1), True)
    
    # Draw horizontal line
    draw_line(grid, (y2, x1), (y2, x2), False)

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int], is_vertical: bool):
    """Draw a line from start to end on the grid, avoiding overwriting red dots."""
    y1, x1 = start
    y2, x2 = end
    
    if is_vertical:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if grid.values[y][x1] == 0:  # Only fill black cells
                grid.values[y][x1] = 3  # Green
    else:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if grid.values[y1][x] == 0:  # Only fill black cells
                grid.values[y1][x] = 3  # Green
