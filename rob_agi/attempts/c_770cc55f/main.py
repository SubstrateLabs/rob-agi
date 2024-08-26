from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple

def solve_770cc55f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored lines with a yellow rectangle.
    
    1. Identifies top and bottom colored lines and the red line.
    2. Finds overlapping columns between top and bottom lines.
    3. If overlap exists and a red line is present, creates a yellow rectangle:
       - Width is exactly 2 columns, positioned at the rightmost overlap.
       - Extends from just below the red line to just above the bottom of the grid.
    4. Returns the modified grid with the yellow rectangle added, or the original grid if conditions aren't met.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find top and bottom colored lines
    top_line = get_colored_line(output_grid[0])
    bottom_line = get_colored_line(output_grid[-1])
    
    # Find red line
    red_line_row = None
    for i, row in enumerate(output_grid):
        if all(cell == 2 for cell in row):
            red_line_row = i
            break
    
    # Find overlapping columns
    overlap = top_line.intersection(bottom_line)
    
    if len(overlap) < 1 or red_line_row is None:
        return output_grid
    
    # Determine the position of the yellow rectangle
    right_col = max(overlap)
    left_col = right_col - 1
    
    # Create yellow rectangle
    start_row = red_line_row + 1
    end_row = rows - 1  # Just above the bottom of the grid
    
    for row in range(start_row, end_row):
        output_grid.set_cell(row, left_col, 4)
        output_grid.set_cell(row, right_col, 4)
    
    return output_grid

def get_colored_line(row: List[int]) -> Set[int]:
    """Returns a set of column indices for non-black cells in a row."""
    return set(i for i, cell in enumerate(row) if cell != 0)
