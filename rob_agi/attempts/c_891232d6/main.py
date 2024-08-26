from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_891232d6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a tree-like structure connecting orange (7) shapes.
    
    1. Creates a main "trunk" on the rightmost column with orange squares.
    2. Processes horizontal orange lines, adding sky blue (8), yellow (4), and green (3) squares.
    3. Connects vertical orange lines and isolated orange squares to the structure.
    4. Extends the structure to the top and left edges where appropriate.
    5. Ensures magenta (6) squares are connected to the structure.
    
    The result is a tree-like structure with a right-side trunk and branches,
    maintaining specific color transitions and connections.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find the rightmost column with orange squares
    rightmost_col = max([c for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 7], default=-1)
    
    if rightmost_col == -1:
        return output_grid  # No orange squares, return the input grid
    
    # Create the main trunk
    for r in range(rows):
        if r == 0 or input_grid.get_cell(r, rightmost_col) == 7:
            output_grid.set_cell(r, rightmost_col, 2)  # Red trunk
    
    # Process horizontal orange lines
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 7:
                if c > 0 and input_grid.get_cell(r, c-1) == 7:  # Part of a horizontal line
                    output_grid.set_cell(r, c, 8)  # Sky blue
                    if c > 1:
                        output_grid.set_cell(r, c-1, 4)  # Yellow
                        output_grid.set_cell(r, c-2, 3)  # Green
                    # Connect to the trunk
                    for cc in range(c+1, rightmost_col):
                        output_grid.set_cell(r, cc, 2)  # Red connection
    
    # Connect isolated orange squares and vertical lines
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 7 and output_grid.get_cell(r, c) == 7:
                # Connect to the nearest part of the structure
                for cc in range(c+1, rightmost_col+1):
                    if output_grid.get_cell(r, cc) != 0:
                        for ccc in range(c+1, cc):
                            output_grid.set_cell(r, ccc, 2)  # Red connection
                        break
    
    # Extend structure to the left where appropriate
    left_extension = min([c for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) in [6, 7]], default=0)
    for r in range(rows):
        if any(input_grid.get_cell(r, c) in [6, 7] for c in range(left_extension, cols)):
            for c in range(left_extension, cols):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)  # Red extension
                else:
                    break
    
    # Connect magenta squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 6:
                # Find the nearest part of the structure to connect
                for dc in range(1, cols):
                    if c + dc < cols and output_grid.get_cell(r, c + dc) != 0:
                        for cc in range(c + 1, c + dc):
                            output_grid.set_cell(r, cc, 2)  # Red connection
                        break
                    if c - dc >= 0 and output_grid.get_cell(r, c - dc) != 0:
                        for cc in range(c - dc + 1, c):
                            output_grid.set_cell(r, cc, 2)  # Red connection
                        break
    
    return output_grid
