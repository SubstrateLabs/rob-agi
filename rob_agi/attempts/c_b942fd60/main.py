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
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all non-black squares
    3. Determine the optimal vertical line position(s)
    4. Draw vertical red lines
    5. Connect horizontal lines to colored squares
    6. Clean up unnecessary extensions
    7. Return the modified grid
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Identify non-black squares
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    if not non_black:
        return output_grid  # Return original grid if no non-black squares
    
    # Identify columns with colored squares
    colored_cols = sorted(set(c for _, c in non_black))
    
    # Decide on vertical line placement
    if len(colored_cols) == 1:
        optimal_cols = colored_cols
    else:
        def total_distance(columns):
            return sum(min(abs(c - col) for col in columns) for _, c in non_black)
        
        one_col = min(colored_cols, key=lambda col: total_distance([col]))
        two_cols = min(((c1, c2) for c1 in colored_cols for c2 in colored_cols if c1 < c2),
                       key=lambda cols: total_distance(cols))
        
        optimal_cols = [one_col] if total_distance([one_col]) <= total_distance(two_cols) else list(two_cols)
    
    # Draw vertical red lines
    top_row = min(r for r, _ in non_black)
    bottom_row = max(r for r, _ in non_black)
    for col in optimal_cols:
        for r in range(top_row, bottom_row + 1):
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
