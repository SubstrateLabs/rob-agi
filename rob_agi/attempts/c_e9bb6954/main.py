from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9bb6954(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Identify all 3x3 squares of the same non-zero color in the input grid.
    2. Sort the 3x3 squares by color (ascending) to establish precedence.
    3. For each 3x3 square:
       a. Draw a horizontal line across the entire grid at the square's center row.
       b. Draw a vertical line from the square's center, extending up or down based on its position.
    4. Preserve all original non-zero values from the input grid.
    5. Return the transformed grid with these lines drawn.
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
        return sorted(squares)  # Sort by color (ascending)

    def draw_lines(color, center_row, center_col):
        # Draw horizontal line
        for col in range(cols):
            if output_grid.get_cell(center_row, col) == 0 or output_grid.get_cell(center_row, col) > color:
                output_grid.set_cell(center_row, col, color)
        
        # Determine vertical line direction
        if center_row < rows / 2:
            direction = "down"
        elif center_row > rows / 2:
            direction = "up"
        else:
            direction = "both"
        
        # Draw vertical line
        if direction in ["down", "both"]:
            for row in range(center_row, rows):
                if output_grid.get_cell(row, center_col) == 0 or output_grid.get_cell(row, center_col) > color:
                    output_grid.set_cell(row, center_col, color)
        if direction in ["up", "both"]:
            for row in range(center_row, -1, -1):
                if output_grid.get_cell(row, center_col) == 0 or output_grid.get_cell(row, center_col) > color:
                    output_grid.set_cell(row, center_col, color)

    squares = find_3x3_squares(input_grid)
    for color, center_row, center_col in squares:
        draw_lines(color, center_row, center_col)

    # Preserve original non-zero values
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return output_grid
