from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ce8d95cc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by compressing it while preserving vertical and horizontal lines.
    
    The solution:
    1. Identifies vertical and horizontal lines in the input grid.
    2. Creates a new grid with dimensions based on the number of lines, ensuring at least one column/row of empty space between lines.
    3. Places vertical lines first, preserving their relative positions.
    4. Places horizontal lines, overwriting intersections with vertical lines.
    5. Adds empty columns and rows at the edges and between lines.
    
    This approach maintains the relative positioning and colors of all lines while
    compressing the grid to its essential features, correctly handling intersections
    and preserving the structure and continuity of both horizontal and vertical lines.
    """
    vertical_lines = find_vertical_lines(input_grid)
    horizontal_lines = find_horizontal_lines(input_grid)
    
    new_width = max(2 * len(vertical_lines) + 1, 3)  # Ensure at least 3 columns
    new_height = max(2 * len(horizontal_lines) + 1, 3)  # Ensure at least 3 rows
    
    output_grid = ColoredGrid(values=[[0 for _ in range(new_width)] for _ in range(new_height)])
    
    # Place vertical lines
    for i, (col, color) in enumerate(vertical_lines):
        output_col = 2 * i + 1
        for row in range(new_height):
            output_grid.values[row][output_col] = color
    
    # Place horizontal lines, overwriting intersections
    for i, (row, color) in enumerate(horizontal_lines):
        output_row = 2 * i + 1
        for col in range(new_width):
            output_grid.values[output_row][col] = color
    
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
