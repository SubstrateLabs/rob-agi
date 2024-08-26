from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9bb6954(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Scan the input grid to identify all 3x3 squares of the same color.
    2. Determine line directions based on the position of 3x3 squares.
    3. Sort the 3x3 squares by color and position.
    4. Create an output grid as a copy of the input grid.
    5. Draw lines for each 3x3 square, respecting color precedence and original non-zero values.
    6. Return the transformed grid with these lines drawn.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def find_3x3_squares(grid):
        squares = []
        for r in range(rows - 2):
            for c in range(cols - 2):
                if all(grid.get_cell(r+i, c+j) == grid.get_cell(r, c) != 0 
                       for i in range(3) for j in range(3)):
                    squares.append((r, c, grid.get_cell(r, c)))
        return squares

    def is_vertical(col):
        return col < 3 or col >= cols - 3

    squares = find_3x3_squares(input_grid)
    squares.sort(key=lambda x: (x[2], x[0], x[1]))  # Sort by color, then position

    def draw_line(square, is_vertical):
        r, c, color = square
        if is_vertical:
            center_col = c + 1
            for row in range(rows):
                if output_grid.get_cell(row, center_col) < color and input_grid.get_cell(row, center_col) == 0:
                    output_grid.set_cell(row, center_col, color)
        else:
            center_row = r + 1
            for col in range(cols):
                if output_grid.get_cell(center_row, col) < color and input_grid.get_cell(center_row, col) == 0:
                    output_grid.set_cell(center_row, col, color)

    processed_colors = set()
    for square in squares:
        color = square[2]
        if color not in processed_colors:
            draw_line(square, is_vertical(square[1]))
            processed_colors.add(color)

    return output_grid
