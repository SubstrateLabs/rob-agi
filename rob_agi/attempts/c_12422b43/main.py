from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import math

def solve_12422b43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending vertical patterns in each column.
    
    1. Preserves the first column unchanged.
    2. For each subsequent column:
       a. Extracts the entire pattern, including zeros.
       b. Extends the pattern to fill the entire column height.
       c. Replaces the column with the extended pattern.
    3. Handles cases where the pattern doesn't divide evenly into the column height.
    4. Leaves columns that are entirely zero unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with extended vertical patterns.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    for c in range(1, cols):  # Start from the second column
        pattern = [input_grid.get_cell(r, c) for r in range(rows)]
        
        if any(pattern):  # If the column is not entirely zeros
            repetitions = math.ceil(rows / len(pattern))
            extended_pattern = (pattern * repetitions)[:rows]
            
            for r in range(rows):
                output_grid.set_cell(r, c, extended_pattern[r])
    
    return output_grid
