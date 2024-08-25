from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_12422b43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending vertical patterns downwards.
    
    1. Identifies vertical patterns in each column, starting from the second column.
    2. Extends these patterns downwards, filling all zero (empty) cells.
    3. Preserves existing non-zero values in the grid.
    4. Repeats the pattern as needed to fill the entire column.
    5. Continues the pattern even after encountering non-zero cells.
    6. Leaves the first column unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with extended vertical patterns.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    for c in range(1, cols):  # Start from the second column
        pattern = []
        for r in range(rows):
            if input_grid.get_cell(r, c) != 0:
                pattern.append(input_grid.get_cell(r, c))
        
        if pattern:
            p_index = 0
            for r in range(rows):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, pattern[p_index])
                p_index = (p_index + 1) % len(pattern)
    
    return output_grid
