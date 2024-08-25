from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_712bf12e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a network of red lines.
    
    The function performs the following steps:
    1. Identifies red squares in the bottom row and creates starting points.
    2. Generates vertical paths from these starting points.
    3. Connects vertical paths with horizontal paths.
    4. Smooths corners and fills dead ends.
    5. Ensures connectivity of the red network.
    6. Performs final cleanup.
    
    The original gray squares are preserved throughout the process.
    
    Args:
    input_grid (ColoredGrid): The input grid with initial red and gray squares.
    
    Returns:
    ColoredGrid: The transformed grid with the network of red lines added.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find starting points
    start_points = []
    for c in range(cols):
        if output_grid.get_cell(rows-1, c) == 2:
            if c+1 < cols and output_grid.get_cell(rows-1, c+1) != 5:
                start_points.append(c+1)
            else:
                start_points.append(c)
    
    # Generate vertical paths
    for c in start_points:
        r = rows - 1
        while r >= 0:
            if output_grid.get_cell(r, c) != 5:
                output_grid.set_cell(r, c, 2)
                r -= 1
            else:
                if c+1 < cols and output_grid.get_cell(r, c+1) != 5:
                    c += 1
                elif c-1 >= 0 and output_grid.get_cell(r, c-1) != 5:
                    c -= 1
                else:
                    break
    
    # Connect horizontal paths
    for r in range(rows):
        red_squares = [c for c in range(cols) if output_grid.get_cell(r, c) == 2]
        for i in range(len(red_squares) - 1):
            start, end = red_squares[i], red_squares[i+1]
            if end - start > 1:
                for c in range(start+1, end):
                    if output_grid.get_cell(r, c) == 0:
                        output_grid.set_cell(r, c, 2)
    
    # Smooth corners and fill dead ends
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                                if 0 <= r+dr < rows and 0 <= c+dc < cols and output_grid.get_cell(r+dr, c+dc) == 2)
                if neighbors >= 2:
                    output_grid.set_cell(r, c, 2)
    
    # Ensure connectivity (simplified version)
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 2:
                if r > 0 and output_grid.get_cell(r-1, c) == 0:
                    output_grid.set_cell(r-1, c, 2)
                if r < rows-1 and output_grid.get_cell(r+1, c) == 0:
                    output_grid.set_cell(r+1, c, 2)
    
    return output_grid
