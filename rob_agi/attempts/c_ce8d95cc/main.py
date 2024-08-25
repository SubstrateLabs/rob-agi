from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ce8d95cc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by compressing it while preserving vertical and horizontal lines.
    
    The solution:
    1. Identifies vertical and horizontal lines in the input grid.
    2. Creates a new grid with dimensions based on the number of lines.
    3. Places vertical lines in odd-numbered columns.
    4. Places horizontal lines in odd-numbered rows, preserving intersections with vertical lines.
    5. Handles thick lines on the edges by filling adjacent columns/rows.
    
    This approach maintains the relative positioning and colors of all lines while
    compressing the grid to its essential features, correctly handling intersections.
    """
    vertical_lines = find_vertical_lines(input_grid)
    horizontal_lines = find_horizontal_lines(input_grid)
    
    new_width = 2 * len(vertical_lines) + 1
    new_height = 2 * len(horizontal_lines) + 1
    
    output_grid = ColoredGrid(values=[[0 for _ in range(new_width)] for _ in range(new_height)])
    
    # Place vertical lines
    for i, (col, color, is_thick) in enumerate(vertical_lines):
        output_col = 2 * i + 1
        for row in range(new_height):
            output_grid.values[row][output_col] = color
        if is_thick:
            if i == 0:  # Left edge
                for row in range(new_height):
                    output_grid.values[row][0] = color
            elif i == len(vertical_lines) - 1:  # Right edge
                for row in range(new_height):
                    output_grid.values[row][-1] = color
    
    # Place horizontal lines
    for i, (row, color, is_thick) in enumerate(horizontal_lines):
        output_row = 2 * i + 1
        for col in range(new_width):
            if col % 2 == 1:  # Odd-numbered columns (vertical lines)
                vertical_color = output_grid.values[output_row][col]
                if vertical_color != 0:
                    output_grid.values[output_row][col] = vertical_color
                else:
                    output_grid.values[output_row][col] = color
            else:
                output_grid.values[output_row][col] = color
        if is_thick:
            if i == 0:  # Top edge
                output_grid.values[0] = [color if c % 2 == 0 else output_grid.values[0][c] for c in range(new_width)]
            elif i == len(horizontal_lines) - 1:  # Bottom edge
                output_grid.values[-1] = [color if c % 2 == 0 else output_grid.values[-1][c] for c in range(new_width)]
    
    return output_grid

def find_vertical_lines(grid: ColoredGrid) -> List[Tuple[int, int, bool]]:
    """Find vertical lines in the grid, returning (column, color, is_thick) tuples."""
    lines = []
    rows, cols = grid.get_dimensions()
    for col in range(cols):
        if all(grid.values[row][col] != 0 for row in range(rows)):
            color = grid.values[0][col]
            is_thick = (col > 0 and grid.values[0][col-1] == color) or (col < cols-1 and grid.values[0][col+1] == color)
            lines.append((col, color, is_thick))
    return lines

def find_horizontal_lines(grid: ColoredGrid) -> List[Tuple[int, int, bool]]:
    """Find horizontal lines in the grid, returning (row, color, is_thick) tuples."""
    lines = []
    rows, cols = grid.get_dimensions()
    for row in range(rows):
        if all(grid.values[row][col] != 0 for col in range(cols)):
            color = grid.values[row][0]
            is_thick = (row > 0 and grid.values[row-1][0] == color) or (row < rows-1 and grid.values[row+1][0] == color)
            lines.append((row, color, is_thick))
    return lines
