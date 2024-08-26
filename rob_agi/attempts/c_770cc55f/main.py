from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple

def solve_770cc55f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored lines with a yellow rectangle.
    
    1. Identifies top and bottom colored lines and the red line.
    2. Finds overlapping columns between top and bottom lines.
    3. If overlap exists, creates a yellow rectangle connecting either top or bottom line to the red line,
       choosing the connection that results in the larger area.
    4. Returns the modified grid with the yellow rectangle added, or the original grid if no overlap.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find top and bottom colored lines
    top_line = get_colored_line(output_grid[0])
    bottom_line = get_colored_line(output_grid[-1])
    
    # Find red line
    red_line_row = next(i for i, row in enumerate(output_grid) if all(cell == 2 for cell in row))
    
    # Find overlapping columns
    overlap = top_line.intersection(bottom_line)
    
    if not overlap:
        return output_grid
    
    # Determine rectangle dimensions
    left_col, right_col = min(overlap), max(overlap)
    width = right_col - left_col + 1
    
    # Decide whether to connect to top or bottom
    top_area = width * red_line_row
    bottom_area = width * (rows - red_line_row - 1)
    
    if top_area >= bottom_area:
        start_row, end_row = 1, red_line_row
    else:
        start_row, end_row = red_line_row + 1, rows - 1
    
    # Create yellow rectangle
    for row in range(start_row, end_row):
        for col in range(left_col, right_col + 1):
            output_grid[row][col] = 4
    
    return output_grid

def get_colored_line(row: List[int]) -> Set[int]:
    """Returns a set of column indices for non-black cells in a row."""
    return set(i for i, cell in enumerate(row) if cell != 0)
