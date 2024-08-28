from rob_agi.colored_grid import ColoredGrid
from typing import List

def get_column_pattern(grid: ColoredGrid, column_index: int) -> List[int]:
    """Extract the non-zero pattern from the top of the column."""
    pattern = []
    for r in range(grid.num_rows):
        cell = grid.get_cell(r, column_index)
        if cell != 0:
            pattern.append(cell)
        elif pattern:  # Stop if we've found a non-zero pattern and hit a zero
            break
    return pattern

def extend_pattern(pattern: List[int], column_height: int) -> List[int]:
    """Extend the pattern to fill the column height."""
    if not pattern:
        return [0] * column_height
    repetitions = (column_height + len(pattern) - 1) // len(pattern)
    extended_pattern = (pattern * repetitions)[:column_height]
    return extended_pattern

def solve_12422b43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending vertical patterns in each column.
    
    1. Preserves the first column unchanged.
    2. For each subsequent column:
       a. Extracts the non-zero pattern from the top of the column.
       b. If a non-zero pattern exists, extends it to fill the entire column height.
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
        pattern = get_column_pattern(input_grid, c)
        extended_pattern = extend_pattern(pattern, rows)
        
        for r in range(rows):
            output_grid.set_cell(r, c, extended_pattern[r])
    
    return output_grid
