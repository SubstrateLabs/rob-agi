from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9bb6954(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Identify all 3x3 squares of the same non-zero color in the input grid.
    2. Determine the leftmost and rightmost columns containing 3x3 squares.
    3. Draw vertical lines using the leftmost and rightmost columns' colors.
    4. Draw horizontal lines for each unique center row of 3x3 squares using their colors.
    5. When drawing lines, only overwrite cells with higher-numbered colors or zeros.
    6. Preserve all original non-zero values from the input grid.
    7. Return the transformed grid with these lines drawn.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_3x3_square(r, c, color):
        if r + 2 >= rows or c + 2 >= cols:
            return False
        return all(input_grid.get_cell(r+i, c+j) == color for i in range(3) for j in range(3))

    def find_3x3_squares():
        squares = []
        for r in range(rows - 2):
            for c in range(cols - 2):
                color = input_grid.get_cell(r, c)
                if color != 0 and is_3x3_square(r, c, color):
                    squares.append((c+1, r+1, color))  # (center_col, center_row, color)
        return squares

    def draw_line(start, end, color):
        sr, sc = start
        er, ec = end
        for r in range(sr, er + 1):
            for c in range(sc, ec + 1):
                if output_grid.get_cell(r, c) == 0 or output_grid.get_cell(r, c) > color:
                    output_grid.set_cell(r, c, color)

    squares = find_3x3_squares()
    
    if squares:
        left_col = min(square[0] for square in squares)
        right_col = max(square[0] for square in squares)
        left_color = next(square[2] for square in squares if square[0] == left_col)
        right_color = next(square[2] for square in squares if square[0] == right_col)
        
        # Draw vertical lines
        draw_line((0, left_col), (rows - 1, left_col), left_color)
        draw_line((0, right_col), (rows - 1, right_col), right_color)
        
        # Draw horizontal lines
        for _, center_row, color in sorted(squares, key=lambda x: x[1]):
            draw_line((center_row, 0), (center_row, cols - 1), color)

    # Preserve original non-zero values
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return output_grid
