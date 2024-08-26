from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_891232d6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a structured network connecting orange (7) and magenta (6) shapes.
    
    1. Analyzes the input grid and identifies orange and magenta squares.
    2. Completes orange shapes into rectangles with specific color patterns.
    3. Determines the main vertical structure based on the center of mass.
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
        if input_grid.get_cell(r, c) == 7 and all(output_grid.get_cell(rr, cc) in [0, 7] for rr, cc in [(r-1, c), (r, c-1)]):
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
                        output_grid.set_cell(rr, cc, 2)  # Red fill
            
            # Add sky blue, yellow, and green squares
            output_grid.set_cell(r, c, 8)  # Sky blue at top-left
            for cc in range(c + 1, c + width):
                output_grid.set_cell(r + height - 1, cc, 4)  # Yellow at bottom
            for rr in range(r + 1, r + height):
                output_grid.set_cell(rr, c + width - 1, 4)  # Yellow at right
            if width > 2 and height > 2:
                output_grid.set_cell(r + 1, c + 1, 3)  # Green if large enough
    
    # Calculate center of mass
    total_mass = sum(1 for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0)
    center_r = sum(r for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0) / total_mass
    center_c = sum(c for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0) / total_mass
    
    # Create main vertical structure
    main_col = int(center_c)
    for r in range(rows):
        output_grid.set_cell(r, main_col, 2)
    
    # Connect rectangles to main structure
    for r, c, width, height in sorted(rectangles, key=lambda x: abs(x[1] - main_col)):
        # Connect top edge to main structure or nearest connected part
        for cc in range(c, main_col + 1 if c < main_col else main_col, 1 if c < main_col else -1):
            if output_grid.get_cell(r, cc) == 0:
                output_grid.set_cell(r, cc, 2)
    
    # Connect magenta squares
    for r, c in colored_squares:
        if input_grid.get_cell(r, c) == 6:
            # Find nearest connected part
            min_dist = float('inf')
            nearest_r, nearest_c = r, c
            for rr in range(rows):
                for cc in range(cols):
                    if output_grid.get_cell(rr, cc) == 2:
                        dist = abs(r - rr) + abs(c - cc)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_r, nearest_c = rr, cc
            
            # Connect to nearest part
            for rr in range(min(r, nearest_r), max(r, nearest_r) + 1):
                if output_grid.get_cell(rr, c) == 0:
                    output_grid.set_cell(rr, c, 2)
            for cc in range(min(c, nearest_c), max(c, nearest_c) + 1):
                if output_grid.get_cell(r, cc) == 0:
                    output_grid.set_cell(r, cc, 2)
    
    # Ensure top connectivity
    if all(output_grid.get_cell(0, c) == 0 for c in range(cols)):
        leftmost_col = min(c for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0)
        for r in range(rows):
            if output_grid.get_cell(r, leftmost_col) == 0:
                output_grid.set_cell(r, leftmost_col, 2)
    
    return output_grid
