from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a partial frame or network of red lines that:
    1. Connects non-black squares horizontally and vertically
    2. Preserves the original positions and colors of non-black squares
    3. Ensures red lines don't extend beyond the last colored square in any direction
    4. Handles both simple and complex grid configurations
    
    Steps:
    1. Initialize a copy of the input grid
    2. Identify non-black squares
    3. Process horizontal lines
    4. Process vertical lines
    5. Connect isolated squares
    6. Clean up any stray red cells
    7. Return the modified grid
    """
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Identify non-black squares
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    # Process horizontal lines
    for r in range(rows):
        row_squares = [c for c in range(cols) if (r, c) in non_black]
        if len(row_squares) >= 2:
            for c in range(row_squares[0], row_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Process vertical lines
    for c in range(cols):
        col_squares = [r for r in range(rows) if (r, c) in non_black]
        if len(col_squares) >= 2:
            for r in range(col_squares[0], col_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Connect isolated squares
    for r, c in non_black:
        # Extend horizontal line
        left = right = c
        while left > 0 and output_grid.get_cell(r, left - 1) == 0:
            left -= 1
            output_grid.set_cell(r, left, 2)
        while right < cols - 1 and output_grid.get_cell(r, right + 1) == 0:
            right += 1
            output_grid.set_cell(r, right, 2)
        
        # Extend vertical line
        top = bottom = r
        while top > 0 and output_grid.get_cell(top - 1, c) == 0:
            top -= 1
            output_grid.set_cell(top, c, 2)
        while bottom < rows - 1 and output_grid.get_cell(bottom + 1, c) == 0:
            bottom += 1
            output_grid.set_cell(bottom, c, 2)
    
    # Clean up stray red cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 2:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and output_grid.get_cell(r + dr, c + dc) == 2)
                if neighbors == 0:
                    output_grid.set_cell(r, c, 0)
    
    return output_grid
