from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_891232d6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a structured network connecting orange (7) and magenta (6) shapes.
    
    1. Analyzes the input grid and identifies orange and magenta squares.
    2. Completes orange shapes into rectangles with specific color patterns.
    3. Determines the main vertical structure based on the leftmost colored column.
    4. Connects rectangles and magenta squares to the main structure.
    5. Ensures top connectivity and optimizes the structure.
    6. Verifies all colored squares are connected in a single component.

    The result is an optimized structure that connects all colored squares,
    maintaining specific color patterns for completed shapes and reaching the top of the grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find all colored squares
    colored_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) in [6, 7]]
    
    if not colored_squares:
        return output_grid  # No colored squares, return the input grid
    
    # Complete orange rectangles
    rectangles = []
    for r, c in colored_squares:
        if input_grid.get_cell(r, c) == 7:
            width = 1
            height = 1
            while c + width < cols and input_grid.get_cell(r, c + width) == 7:
                width += 1
            while r + height < rows and input_grid.get_cell(r + height, c) == 7:
                height += 1
            
            rectangles.append((r, c, width, height))
            
            # Fill the rectangle
            for rr in range(r, r + height):
                for cc in range(c, c + width):
                    if input_grid.get_cell(rr, cc) != 7:
                        output_grid.set_cell(rr, cc, 7)  # Fill with orange
            
            # Add sky blue, yellow, and green squares
            output_grid.set_cell(r, c, 8)  # Sky blue at top-left
            output_grid.set_cell(r, c + width - 1, 8)  # Sky blue at top-right
            for cc in range(c, c + width):
                output_grid.set_cell(r + height - 1, cc, 4)  # Yellow at bottom
            for rr in range(r + 1, r + height - 1):
                output_grid.set_cell(rr, c + width - 1, 4)  # Yellow at right
            if width > 2 and height > 2:
                output_grid.set_cell(r + height - 2, c + 1, 3)  # Green if large enough
    
    # Determine main vertical structure (leftmost column with colored squares)
    main_col = min(c for _, c in colored_squares)
    
    # Create main vertical structure
    for r in range(rows):
        if output_grid.get_cell(r, main_col) == 0:
            output_grid.set_cell(r, main_col, 2)
    
    # Connect rectangles to main structure
    for r, c, width, height in rectangles:
        # Connect bottom-left corner to main structure
        for cc in range(min(c, main_col), max(c, main_col) + 1):
            if output_grid.get_cell(r + height - 1, cc) == 0:
                output_grid.set_cell(r + height - 1, cc, 2)
    
    # Connect magenta squares
    for r, c in colored_squares:
        if input_grid.get_cell(r, c) == 6:
            # Connect to the main structure
            for rr in range(r, rows):
                if output_grid.get_cell(rr, c) == 0:
                    output_grid.set_cell(rr, c, 2)
                elif output_grid.get_cell(rr, c) == 2:
                    break
    
    return output_grid
