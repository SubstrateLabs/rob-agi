from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a minimal network of red lines that:
    1. Connects all non-black squares horizontally and vertically
    2. Preserves the original positions and colors of non-black squares
    3. Ensures red lines don't extend beyond the last colored square in any direction
    4. Handles both simple and complex grid configurations
    5. Cleans up any unnecessary or stray red lines
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all non-black squares
    3. Process horizontal connections
    4. Process vertical connections
    5. Connect isolated squares
    6. Clean up unnecessary red lines
    7. Return the modified grid
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    # Process horizontal connections
    for r in range(rows):
        row_squares = [c for c in range(cols) if (r, c) in non_black]
        if len(row_squares) >= 2:
            for c in range(row_squares[0], row_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Process vertical connections
    for c in range(cols):
        col_squares = [r for r in range(rows) if (r, c) in non_black]
        if len(col_squares) >= 2:
            for r in range(col_squares[0], col_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Connect isolated squares
    for r, c in non_black:
        if all(output_grid.get_cell(r + dr, c + dc) == 0 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= r + dr < rows and 0 <= c + dc < cols):
            # Extend vertical line
            for nr in range(r - 1, -1, -1):
                if output_grid.get_cell(nr, c) != 0:
                    break
                output_grid.set_cell(nr, c, 2)
            for nr in range(r + 1, rows):
                if output_grid.get_cell(nr, c) != 0:
                    break
                output_grid.set_cell(nr, c, 2)
            
            # Extend horizontal line
            for nc in range(c - 1, -1, -1):
                if output_grid.get_cell(r, nc) != 0:
                    break
                output_grid.set_cell(r, nc, 2)
            for nc in range(c + 1, cols):
                if output_grid.get_cell(r, nc) != 0:
                    break
                output_grid.set_cell(r, nc, 2)
    
    # Clean up unnecessary red lines
    changes = True
    while changes:
        changes = False
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 2:
                    neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= r + dr < rows and 0 <= c + dc < cols and output_grid.get_cell(r + dr, c + dc) != 0)
                    if neighbors < 2:
                        output_grid.set_cell(r, c, 0)
                        changes = True
    
    return output_grid
