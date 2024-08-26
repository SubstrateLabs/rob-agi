from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Determines if the line should be horizontal or vertical based on the spread of colored squares.
    2. For horizontal lines, draws in each row where the color appears, from leftmost to rightmost occurrence.
    3. For vertical lines, draws in the leftmost column where the color appears, from topmost to bottommost occurrence.
    4. Preserves intersections by not overwriting existing colors.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    color_positions = {}

    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color != 0:  # Not black
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((row, col))

    for color, positions in color_positions.items():
        cols, rows = zip(*positions)
        horizontal_spread = max(cols) - min(cols)
        vertical_spread = max(rows) - min(rows)

        if horizontal_spread >= vertical_spread:
            row_groups = {}
            for row, col in positions:
                if row not in row_groups:
                    row_groups[row] = []
                row_groups[row].append(col)
            
            for row, cols in row_groups.items():
                left, right = min(cols), max(cols)
                for col in range(left, right + 1):
                    if output_grid.values[row][col] == 0:
                        output_grid.values[row][col] = color
        else:
            left_col = min(cols)
            top, bottom = min(rows), max(rows)
            for row in range(top, bottom + 1):
                if output_grid.values[row][left_col] == 0:
                    output_grid.values[row][left_col] = color

    return output_grid
