from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9bb6954(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Identify all 3x3 squares of the same non-zero color in the input grid.
    2. Sort the 3x3 squares by color (ascending) and then by column position (left to right).
    3. Draw horizontal lines for each 3x3 square's center row.
    4. Draw two vertical lines at fixed positions (1/3 and 2/3 of grid width).
    5. Use the leftmost 3x3 square's color for the left vertical line and the rightmost for the right.
    6. When drawing lines, only overwrite cells with higher-numbered colors or zeros.
    7. Preserve all original non-zero values from the input grid.
    8. Return the transformed grid with these lines drawn.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def find_3x3_squares(grid):
        squares = []
        for r in range(rows - 2):
            for c in range(cols - 2):
                if all(grid.get_cell(r+i, c+j) == grid.get_cell(r, c) != 0 
                       for i in range(3) for j in range(3)):
                    squares.append((grid.get_cell(r, c), r+1, c+1))  # (color, center_row, center_col)
        return sorted(squares, key=lambda x: (x[0], x[2]))  # Sort by color, then by column

    def draw_horizontal_line(color, row):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 0 or output_grid.get_cell(row, col) > color:
                output_grid.set_cell(row, col, color)

    def draw_vertical_line(color, col):
        for row in range(rows):
            if output_grid.get_cell(row, col) == 0 or output_grid.get_cell(row, col) > color:
                output_grid.set_cell(row, col, color)

    squares = find_3x3_squares(input_grid)
    
    if squares:
        left_color = squares[0][0]
        right_color = squares[-1][0]
        
        # Draw horizontal lines
        for color, center_row, _ in squares:
            draw_horizontal_line(color, center_row)
        
        # Draw vertical lines
        left_col = cols // 3
        right_col = (2 * cols) // 3
        draw_vertical_line(left_color, left_col)
        draw_vertical_line(right_color, right_col)

    # Preserve original non-zero values
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return output_grid
