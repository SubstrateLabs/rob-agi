from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors vertically based on the following rules:
    1. Process each column independently.
    2. For each column, create vertical lines for each color, prioritizing longer lines.
    3. Extend colors upwards and downwards until hitting another color or the grid edge.
    4. Connect distant dots of the same color within a column.
    5. Preserve original non-black cells in their positions.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    def process_column(col):
        colors = [(row, input_grid.values[row][col]) for row in range(height) if input_grid.values[row][col] != 0]
        if not colors:
            return

        # Calculate potential line lengths
        color_lengths = {}
        for i, (row, color) in enumerate(colors):
            top = row
            bottom = row
            for j in range(i-1, -1, -1):
                if colors[j][1] != color:
                    break
                top = colors[j][0]
            for j in range(i+1, len(colors)):
                if colors[j][1] != color:
                    break
                bottom = colors[j][0]
            length = bottom - top + 1
            if color not in color_lengths or length > color_lengths[color][1]:
                color_lengths[color] = (top, length)

        # Sort colors by potential line length
        sorted_colors = sorted(color_lengths.items(), key=lambda x: x[1][1], reverse=True)

        # Fill the column
        for color, (top, _) in sorted_colors:
            start = top
            end = top
            for row, c in colors:
                if c == color:
                    start = min(start, row)
                    end = max(end, row)
            for row in range(start, end + 1):
                if new_grid.values[row][col] == 0:
                    new_grid.values[row][col] = color

    # Process each column
    for col in range(width):
        process_column(col)

    # Preserve original colors
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] != 0:
                new_grid.values[row][col] = input_grid.values[row][col]

    return new_grid
