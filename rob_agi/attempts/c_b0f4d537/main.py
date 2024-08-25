from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b0f4d537(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies a vertical dividing line (usually color 5) in the input grid.
    2. Finds horizontal lines of single colors in the input grid.
    3. Creates a new 7-column wide grid with the same height as the input.
    4. Places the vertical dividing line color in the 4th column of the new grid.
    5. Fills in horizontal lines in the new grid based on the input grid.
    6. For non-horizontal lines, places the first non-zero color to the right of the dividing line in the 3rd column,
       and the second non-zero color (if it exists) in the 5th column of the new grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    def find_vertical_line(grid: ColoredGrid) -> Tuple[int, int]:
        rows, cols = grid.get_dimensions()
        for c in range(cols):
            column = [grid.get_cell(r, c) for r in range(rows)]
            if len(set(column)) == 2 and 0 in column and sum(column) > 0:
                return c, max(set(column))
        return -1, -1

    def find_horizontal_lines(grid: ColoredGrid) -> List[Tuple[int, int]]:
        rows, cols = grid.get_dimensions()
        lines = []
        for r in range(rows):
            row = [grid.get_cell(r, c) for c in range(cols)]
            if len(set(row)) == 1 and row[0] != 0:
                lines.append((r, row[0]))
        return lines

    rows, cols = input_grid.get_dimensions()
    vertical_line_col, vertical_line_color = find_vertical_line(input_grid)
    horizontal_lines = find_horizontal_lines(input_grid)

    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(rows)])

    for r in range(rows):
        if r in [row for row, _ in horizontal_lines]:
            # Set horizontal lines
            color = next(color for row, color in horizontal_lines if row == r)
            for c in range(7):
                output_grid.set_cell(r, c, color)
        else:
            # Set the vertical line
            output_grid.set_cell(r, 3, vertical_line_color)
            
            # Find colors to the right of the dividing line
            right_colors = [input_grid.get_cell(r, c) for c in range(vertical_line_col + 1, cols) if input_grid.get_cell(r, c) != 0]
            
            if len(right_colors) >= 1:
                output_grid.set_cell(r, 2, right_colors[0])
                output_grid.set_cell(r, 4, right_colors[-1])  # Use the last color if there's only one

    return output_grid
