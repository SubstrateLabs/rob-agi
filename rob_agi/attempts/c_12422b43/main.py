from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_12422b43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending vertical patterns downwards.
    
    1. Identifies vertical patterns in the input grid, excluding the leftmost column.
    2. Extends these patterns downwards to fill the grid.
    3. Preserves the original content of the first two rows.
    4. If a pattern is blocked by an existing color, it stops at that point.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with extended vertical patterns.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Identify and extend vertical patterns
    for c in range(1, cols):  # Start from the second column
        pattern = []
        for r in range(2, rows):  # Start from the third row
            if input_grid.get_cell(r, c) != 0:
                pattern.append(input_grid.get_cell(r, c))
            elif pattern:
                break
        
        if pattern:
            r = 2  # Start filling from the third row
            while r < rows:
                for color in pattern:
                    if r < rows and output_grid.get_cell(r, c) == 0:
                        output_grid.set_cell(r, c, color)
                        r += 1
                    else:
                        break
                if r >= rows or output_grid.get_cell(r, c) != 0:
                    break
    
    return output_grid
