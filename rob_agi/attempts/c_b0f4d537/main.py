from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b0f4d537(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies a vertical dividing line (usually color 5) in the input grid.
    2. Determines the color for vertical line(s) in the output from the left side of the input.
    3. Determines the positions for vertical line(s) in the output from the right side of the input.
    4. Creates a new 7-column wide grid with the same height as the input.
    5. Draws vertical line(s) in the output grid.
    6. Processes each row:
       - If it's a horizontal line in the input, fills it in the output between vertical lines.
       - Otherwise, places non-zero colors from the right side of the input in corresponding positions.
    7. Fills remaining cells with black (0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    def find_vertical_line(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        for c in range(cols):
            column = [grid.get_cell(r, c) for r in range(rows)]
            if len(set(column)) == 2 and 0 in column and sum(column) > 0:
                return c
        return -1

    def find_vertical_line_color(grid: ColoredGrid, dividing_line: int) -> int:
        rows, _ = grid.get_dimensions()
        for c in range(dividing_line):
            for r in range(rows):
                color = grid.get_cell(r, c)
                if color != 0:
                    return color
        return 1  # Default to blue if no color found

    def find_output_line_positions(grid: ColoredGrid, dividing_line: int) -> List[int]:
        rows, cols = grid.get_dimensions()
        colors = set()
        for c in range(dividing_line + 1, cols):
            for r in range(rows):
                color = grid.get_cell(r, c)
                if color != 0:
                    colors.add(color)
                    if len(colors) == 2:
                        return [3, 5]
        return [4]  # Default to middle column if only one color found

    rows, _ = input_grid.get_dimensions()
    dividing_line = find_vertical_line(input_grid)
    vertical_line_color = find_vertical_line_color(input_grid, dividing_line)
    output_line_positions = find_output_line_positions(input_grid, dividing_line)

    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(rows)])

    # Draw vertical lines
    for r in range(rows):
        for pos in output_line_positions:
            output_grid.set_cell(r, pos, vertical_line_color)

    for r in range(rows):
        input_row = [input_grid.get_cell(r, c) for c in range(dividing_line)]
        if len(set(input_row)) == 1 and input_row[0] != 0:
            # Horizontal line
            for c in range(min(output_line_positions), max(output_line_positions) + 1):
                output_grid.set_cell(r, c, input_row[0])
        else:
            # Place non-zero colors from right side
            right_colors = [input_grid.get_cell(r, c) for c in range(dividing_line + 1, input_grid.get_dimensions()[1]) if input_grid.get_cell(r, c) != 0]
            for i, pos in enumerate(output_line_positions):
                if i < len(right_colors):
                    output_grid.set_cell(r, pos, right_colors[i])

    return output_grid
