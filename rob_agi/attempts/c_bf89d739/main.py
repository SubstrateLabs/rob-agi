from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function creates a central vertical spine in the leftmost column containing a red dot
    after the first red dot. It then connects all red dots to this spine using horizontal
    green lines. The algorithm ensures that all red dots are connected in a tree-like structure
    with a main vertical trunk.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming a tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid.values[r][c] == 2]
    
    if not red_dots:
        return result_grid

    # Find the spine column (leftmost column with a red dot after the first red dot)
    spine_col = min(c for _, c in red_dots[1:]) if len(red_dots) > 1 else red_dots[0][1]

    # Create the vertical spine
    top_spine = min(r for r, c in red_dots if c == spine_col)
    bottom_spine = max(r for r, c in red_dots if c == spine_col)
    draw_line(result_grid, (top_spine, spine_col), (bottom_spine, spine_col), True)

    # Connect dots to the spine
    for r, c in red_dots:
        if c != spine_col:
            # Extend spine if necessary
            if r < top_spine:
                draw_line(result_grid, (r, spine_col), (top_spine, spine_col), True)
                top_spine = r
            elif r > bottom_spine:
                draw_line(result_grid, (bottom_spine, spine_col), (r, spine_col), True)
                bottom_spine = r
            
            # Draw horizontal line
            draw_line(result_grid, (r, c), (r, spine_col), False)

    return result_grid

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
