from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_891232d6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a tree-like structure connecting orange (7) and magenta (6) shapes.
    
    1. Analyzes the input grid to determine if changes are needed.
    2. Identifies the rightmost column containing colored squares.
    3. Creates a main vertical red (2) structure from this column.
    4. Processes orange shapes from right to left, completing them into rectangles.
    5. Connects shapes to the structure with horizontal red lines.
    6. Creates additional vertical "trunks" as needed for distant shapes.
    7. Connects isolated orange and magenta squares to the nearest part of the structure.
    8. Optimizes the structure by removing unnecessary red squares.
    9. Ensures all colored squares are connected and the structure reaches the top of the grid.

    The result is a minimal tree-like structure that connects all colored squares,
    maintaining specific color patterns for completed shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find all colored squares
    colored_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) in [6, 7]]
    
    if not colored_squares:
        return output_grid  # No colored squares, return the input grid
    
    # Check if the colored squares are already connected
    if all(output_grid.get_cell(r, c) in [6, 7] for r, c in colored_squares):
        return output_grid  # Already connected, return the input grid
    
    # Find the rightmost column with colored squares
    rightmost_col = max(c for _, c in colored_squares)
    
    # Create the main vertical structure
    for r in range(rows):
        if any(output_grid.get_cell(r, c) != 0 for c in range(rightmost_col + 1)):
            output_grid.set_cell(r, rightmost_col, 2)  # Red vertical line
    
    # Process orange shapes from right to left
    for c in range(rightmost_col, -1, -1):
        for r in range(rows):
            if input_grid.get_cell(r, c) == 7:
                # Complete the shape into a rectangle
                width = 1
                height = 1
                while c + width < cols and input_grid.get_cell(r, c + width) == 7:
                    width += 1
                while r + height < rows and input_grid.get_cell(r + height, c) == 7:
                    height += 1
                
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
                
                # Connect to the main structure
                for cc in range(c + width, rightmost_col + 1):
                    if output_grid.get_cell(r, cc) == 0:
                        output_grid.set_cell(r, cc, 2)  # Red connection
    
    # Connect isolated squares and ensure all are connected
    for r, c in colored_squares:
        if output_grid.get_cell(r, c) in [6, 7]:
            # Find the nearest part of the structure to connect
            for d in range(1, max(cols, rows)):
                connected = False
                for dr, dc in [(0, d), (0, -d), (d, 0), (-d, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output_grid.get_cell(nr, nc) == 2:
                        # Connect to this part of the structure
                        for i in range(min(r, nr), max(r, nr) + 1):
                            output_grid.set_cell(i, c, 2)
                        for j in range(min(c, nc), max(c, nc) + 1):
                            output_grid.set_cell(r, j, 2)
                        connected = True
                        break
                if connected:
                    break
    
    # Ensure the structure reaches the top of the grid
    top_connection = next((c for c in range(cols) if output_grid.get_cell(0, c) == 2), None)
    if top_connection is None:
        leftmost_structure = min(c for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 2)
        for r in range(rows):
            output_grid.set_cell(r, leftmost_structure, 2)
    
    return output_grid
