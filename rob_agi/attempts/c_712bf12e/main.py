from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_712bf12e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating vertical red lines and connecting them horizontally.
    
    The function identifies red squares in the bottom row and creates vertical
    red lines in the column immediately to the right of each red square,
    extending from the bottom to the top of the grid. It then connects these
    vertical lines horizontally at various heights, creating a network of red lines.
    The original gray squares and other non-red squares are preserved.
    
    Args:
    input_grid (ColoredGrid): The input grid with initial red and gray squares.
    
    Returns:
    ColoredGrid: The transformed grid with the network of red lines added.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find starting points (red squares in the bottom row)
    start_points = [c for c in range(cols) if output_grid.get_cell(rows-1, c) == 2]
    
    # Create vertical red lines
    for c in start_points:
        new_col = min(c + 1, cols - 1)  # Column immediately to the right, but not exceeding grid bounds
        for r in range(rows):
            if output_grid.get_cell(r, new_col) != 5:  # Don't overwrite gray squares
                output_grid.set_cell(r, new_col, 2)  # Set cell to red (2)
    
    # Connect vertical lines horizontally
    for r in range(rows):
        start = None
        for c in range(cols):
            if output_grid.get_cell(r, c) == 2:
                if start is None:
                    start = c
                else:
                    for cc in range(start + 1, c):
                        if output_grid.get_cell(r, cc) != 5:  # Don't overwrite gray squares
                            output_grid.set_cell(r, cc, 2)
                    start = c
    
    return output_grid
