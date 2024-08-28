from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Starts from the topmost occurrence of the color.
    2. Draws a full vertical line down to the bottom of the grid.
    3. Connects all occurrences horizontally to the main vertical line.
    4. Connects any disconnected squares in the bottom row with a horizontal line.
    5. Preserves original colored squares and intersections.
    6. Uses the minimum number of line segments to connect all squares of the same color.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    unique_colors = set(color for row in input_grid.values for color in row if color != 0)

    for color in unique_colors:
        occurrences = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == color]
        occurrences.sort()  # Sort by row, then column
        top_row, top_col = occurrences[0]

        # Draw main vertical line
        for row in range(top_row, rows):
            if output_grid.values[row][top_col] == 0:
                output_grid.values[row][top_col] = color

        # Connect other occurrences horizontally
        for row, col in occurrences[1:]:
            if col != top_col:
                for c in range(min(col, top_col), max(col, top_col) + 1):
                    if output_grid.values[row][c] == 0:
                        output_grid.values[row][c] = color

        # Connect in the bottom row
        bottom_squares = [c for c in range(cols) if input_grid.values[rows-1][c] == color]
        if bottom_squares:
            left_col, right_col = min(bottom_squares), max(bottom_squares)
            for col in range(left_col, right_col + 1):
                if output_grid.values[rows-1][col] == 0:
                    output_grid.values[rows-1][col] = color

    return output_grid
