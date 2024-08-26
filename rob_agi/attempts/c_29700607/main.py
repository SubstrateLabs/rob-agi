from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Starts from the topmost occurrence of the color.
    2. Draws a full vertical line down to the bottom of the grid.
    3. Connects any disconnected squares in the bottom row with a horizontal line.
    4. Preserves original colored squares and intersections.
    5. Uses the minimum number of line segments to connect all squares of the same color.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    unique_colors = set(color for row in input_grid.values for color in row if color != 0)

    for color in unique_colors:
        top_row, top_col = next((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == color)

        # Draw vertical line
        for row in range(top_row, rows):
            if output_grid.values[row][top_col] == 0:
                output_grid.values[row][top_col] = color

        # Find disconnected squares in bottom row
        bottom_squares = [c for c in range(cols) if input_grid.values[rows-1][c] == color]
        if bottom_squares:
            left_col, right_col = min(bottom_squares), max(bottom_squares)

            # Draw horizontal line in bottom row
            for col in range(min(left_col, top_col), max(right_col, top_col) + 1):
                if output_grid.values[rows-1][col] == 0:
                    output_grid.values[rows-1][col] = color

    return output_grid
