from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_891232d6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a tree-like structure connecting orange (7) and magenta (6) shapes.
    
    1. Identifies the center of mass of orange squares as the starting point.
    2. Creates a main vertical red (2) structure from this point.
    3. Processes horizontal orange lines, adding sky blue (8), yellow (4), and green (3) squares.
    4. Connects isolated orange squares and vertical orange lines to the structure.
    5. Extends the structure to connect all colored squares.
    6. Ensures magenta (6) squares are connected to the structure.
    
    The result is an organic tree-like structure with branches,
    maintaining specific color transitions and connections.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find all orange and magenta squares
    colored_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) in [6, 7]]
    
    if not colored_squares:
        return output_grid  # No colored squares, return the input grid
    
    # Find the center of mass
    center_r = int(sum(r for r, _ in colored_squares) / len(colored_squares))
    center_c = int(sum(c for _, c in colored_squares) / len(colored_squares))
    
    # Create the main vertical structure
    for r in range(rows):
        output_grid.set_cell(r, center_c, 2)  # Red vertical line
    
    # Process horizontal orange lines
    for r in range(rows):
        orange_line = [c for c in range(cols) if input_grid.get_cell(r, c) == 7]
        if len(orange_line) >= 2:
            output_grid.set_cell(r, orange_line[-1], 8)  # Sky blue
            output_grid.set_cell(r, orange_line[-2], 4)  # Yellow
            if len(orange_line) >= 3:
                output_grid.set_cell(r, orange_line[-3], 3)  # Green
            for c in orange_line[:-3]:
                output_grid.set_cell(r, c, 2)  # Red
            # Connect to the main structure
            for c in range(min(orange_line[-1], center_c), max(orange_line[0], center_c) + 1):
                output_grid.set_cell(r, c, 2)
    
    # Connect isolated orange squares and vertical lines
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 7 and output_grid.get_cell(r, c) == 7:
                # Connect horizontally to the nearest part of the structure
                left = right = c
                while left > 0 and output_grid.get_cell(r, left) == 0:
                    left -= 1
                while right < cols - 1 and output_grid.get_cell(r, right) == 0:
                    right += 1
                if output_grid.get_cell(r, left) != 0:
                    for cc in range(left, c + 1):
                        output_grid.set_cell(r, cc, 2)
                elif output_grid.get_cell(r, right) != 0:
                    for cc in range(c, right + 1):
                        output_grid.set_cell(r, cc, 2)
    
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
    
    # Final pass to ensure all colored squares are connected
    for r in range(rows):
        colored_in_row = [c for c in range(cols) if output_grid.get_cell(r, c) != 0]
        if colored_in_row:
            for c in range(min(colored_in_row), max(colored_in_row) + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    return output_grid
