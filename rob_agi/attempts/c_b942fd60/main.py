from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a minimal network of red lines that:
    1. Creates one or two vertical lines based on the distribution of colored squares
    2. Creates horizontal lines to connect colored squares to the vertical line(s)
    3. Preserves the original positions and colors of non-black squares
    4. Ensures red lines don't extend beyond the last colored square in any direction
    5. Handles special cases like single row/column of colored squares
    
    The algorithm optimizes the placement of vertical lines to minimize the total
    distance between colored squares and these lines. It then connects all non-black
    squares to the nearest vertical line with horizontal red lines.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Identify non-black squares
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    if not non_black:
        return output_grid  # Return original grid if no non-black squares
    
    # Identify columns and rows with colored squares
    colored_cols = sorted(set(c for _, c in non_black))
    colored_rows = sorted(set(r for r, _ in non_black))
    
    # Handle special cases
    if len(colored_rows) == 1:  # All colored squares in a single row
        r = colored_rows[0]
        for c in range(min(colored_cols), max(colored_cols) + 1):
            if output_grid.get_cell(r, c) == 0:
                output_grid.set_cell(r, c, 2)
        return output_grid
    
    if len(colored_cols) == 1:  # All colored squares in a single column
        c = colored_cols[0]
        for r in range(min(colored_rows), max(colored_rows) + 1):
            if output_grid.get_cell(r, c) == 0:
                output_grid.set_cell(r, c, 2)
        return output_grid
    
    # Decide on vertical line placement
    def total_distance(columns):
        return sum(min(abs(c - col) for col in columns) for _, c in non_black)
    
    one_col = min(range(cols), key=lambda col: total_distance([col]))
    two_cols = min(((c1, c2) for c1 in range(cols) for c2 in range(c1+1, cols)),
                   key=lambda cols: total_distance(cols))
    
    optimal_cols = list(two_cols) if total_distance(two_cols) < total_distance([one_col]) * 0.8 else [one_col]
    
    # Draw vertical red lines
    for col in optimal_cols:
        for r in range(rows):
            if output_grid.get_cell(r, col) == 0:
                output_grid.set_cell(r, col, 2)
    
    # Connect horizontal lines
    for r, c in non_black:
        if c not in optimal_cols:
            nearest_col = min(optimal_cols, key=lambda col: abs(col - c))
            for cc in range(min(c, nearest_col), max(c, nearest_col) + 1):
                if output_grid.get_cell(r, cc) == 0:
                    output_grid.set_cell(r, cc, 2)
    
    # Clean up unnecessary extensions
    for r in range(rows):
        left = min((c for c in range(cols) if output_grid.get_cell(r, c) != 0), default=cols)
        right = max((c for c in range(cols) if output_grid.get_cell(r, c) != 0), default=-1)
        for c in range(cols):
            if c < left or c > right:
                output_grid.set_cell(r, c, 0)
    
    for c in range(cols):
        top = min((r for r in range(rows) if output_grid.get_cell(r, c) != 0), default=rows)
        bottom = max((r for r in range(rows) if output_grid.get_cell(r, c) != 0), default=-1)
        for r in range(rows):
            if r < top or r > bottom:
                output_grid.set_cell(r, c, 0)
    
    return output_grid
