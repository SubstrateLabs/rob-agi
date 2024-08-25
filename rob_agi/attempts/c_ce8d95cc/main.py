from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ce8d95cc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by compressing it while preserving vertical and horizontal lines.
    
    The solution:
    1. Identifies vertical and horizontal lines in the input grid.
    2. Creates a new grid with dimensions based on the number of lines.
    3. Places horizontal lines first, preserving their full width.
    4. Places vertical lines, respecting intersections with horizontal lines.
    5. Removes empty rows and columns between lines.
    
    This approach maintains the relative positioning and colors of all lines while
    compressing the grid to its essential features, correctly handling intersections
    and preserving the structure and continuity of both horizontal and vertical lines.
    """
    vertical_lines = find_vertical_lines(input_grid)
    horizontal_lines = find_horizontal_lines(input_grid)
    
    new_width = len(vertical_lines) + 1
    new_height = len(horizontal_lines) + 1
    
    output_grid = ColoredGrid(values=[[0 for _ in range(new_width)] for _ in range(new_height)])
    
    # Place horizontal lines first
    for i, (row, color) in enumerate(horizontal_lines):
        output_row = i + 1
        for col in range(new_width):
            output_grid.values[output_row][col] = color
    
    # Place vertical lines, preserving intersections
    for i, (col, color) in enumerate(vertical_lines):
        output_col = i + 1
        for row in range(new_height):
            if output_grid.values[row][output_col] == 0:  # Only fill if not a horizontal line
                output_grid.values[row][output_col] = color
    
    return output_grid

def find_vertical_lines(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find vertical lines in the grid, returning (column, color) tuples."""
    lines = []
    rows, cols = grid.get_dimensions()
    for col in range(cols):
        if all(grid.values[row][col] != 0 for row in range(rows)):
            color = grid.values[0][col]
            lines.append((col, color))
    return lines

def find_horizontal_lines(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find horizontal lines in the grid, returning (row, color) tuples."""
    lines = []
    rows, cols = grid.get_dimensions()
    for row in range(rows):
        if all(grid.values[row][col] != 0 for col in range(cols)):
            color = grid.values[row][0]
            lines.append((row, color))
    return lines
