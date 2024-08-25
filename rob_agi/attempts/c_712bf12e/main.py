from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_712bf12e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a network of red lines.
    
    The function performs the following steps:
    1. Identifies red squares in the bottom row.
    2. Creates vertical red lines in the column immediately to the right of each identified red square.
    3. Connects these vertical lines horizontally at each row.
    4. Extends horizontal lines to the left and right edges when possible.
    5. Fills in any black squares that have red squares both above and below or to the left and right.
    
    The original gray squares are preserved throughout the process.
    
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
    
    # Connect vertical lines horizontally and extend to edges
    for r in range(rows):
        red_squares = [c for c in range(cols) if output_grid.get_cell(r, c) == 2]
        if red_squares:
            # Extend left
            for c in range(red_squares[0]):
                if output_grid.get_cell(r, c) == 5:
                    break
                output_grid.set_cell(r, c, 2)
            
            # Connect between red squares
            for i in range(len(red_squares) - 1):
                for c in range(red_squares[i] + 1, red_squares[i+1]):
                    if output_grid.get_cell(r, c) != 5:
                        output_grid.set_cell(r, c, 2)
            
            # Extend right
            for c in range(red_squares[-1] + 1, cols):
                if output_grid.get_cell(r, c) == 5:
                    break
                output_grid.set_cell(r, c, 2)
    
    # Fill in black squares with red above and below or left and right
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if output_grid.get_cell(r, c) == 0:
                if (output_grid.get_cell(r-1, c) == 2 and output_grid.get_cell(r+1, c) == 2) or \
                   (output_grid.get_cell(r, c-1) == 2 and output_grid.get_cell(r, c+1) == 2):
                    output_grid.set_cell(r, c, 2)
    
    return output_grid
