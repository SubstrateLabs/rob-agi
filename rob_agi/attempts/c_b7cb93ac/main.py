from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import itertools
from typing import List, Tuple

def solve_b7cb93ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3x4 output grid based on color patterns.
    
    1. Analyzes the input grid to identify the three most prominent colors and checks for sky blue.
    2. Identifies key patterns: longest horizontal line, longest vertical line, and corner elements.
    3. Constructs a 3x4 grid with the following layout:
       - Horizontal pattern color fills the middle row.
       - Vertical pattern color fills the leftmost column.
       - Sky blue (if present) or the third most prominent color fills the corners.
    4. Adjusts the grid to ensure all three identified colors are represented.
    5. Ensures consistency: uniform middle row and leftmost column, mirrored top and bottom rows.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    # Step 1: Analyze input grid
    color_counts = Counter(color for row in input_grid.values for color in row if color != 0)
    top_colors = [color for color, _ in color_counts.most_common(3)]
    sky_blue_present = 8 in color_counts

    # Step 2: Identify patterns
    horizontal_color = identify_horizontal_pattern(input_grid)
    vertical_color = identify_vertical_pattern(input_grid)
    corner_color = 8 if sky_blue_present else (set(top_colors) - {horizontal_color, vertical_color}).pop()

    # Step 3 & 4: Construct and adjust the output grid
    output = [
        [vertical_color, horizontal_color, corner_color, corner_color],
        [vertical_color, horizontal_color, horizontal_color, horizontal_color],
        [vertical_color, horizontal_color, corner_color, corner_color]
    ]

    # Step 5: Ensure consistency and all colors are represented
    if len(set(top_colors)) < 3:
        if vertical_color == horizontal_color:
            vertical_color = next(c for c in top_colors if c != horizontal_color)
        if corner_color == horizontal_color or corner_color == vertical_color:
            corner_color = next(c for c in top_colors if c != horizontal_color and c != vertical_color)
    
    output[0][0] = output[2][0] = corner_color
    output[0][1] = output[0][3] = output[2][1] = output[2][3] = horizontal_color

    # Create and return ColoredGrid
    return ColoredGrid(values=output)

def identify_horizontal_pattern(grid: ColoredGrid) -> int:
    """Identify the color of the longest horizontal line."""
    rows, cols = grid.get_dimensions()
    max_length = 0
    max_color = 0
    for row in range(rows):
        for color, group in itertools.groupby(grid.values[row]):
            if color != 0:
                length = len(list(group))
                if length > max_length:
                    max_length = length
                    max_color = color
    return max_color

def identify_vertical_pattern(grid: ColoredGrid) -> int:
    """Identify the color of the longest vertical line."""
    rows, cols = grid.get_dimensions()
    max_length = 0
    max_color = 0
    for col in range(cols):
        column = [grid.values[row][col] for row in range(rows)]
        for color, group in itertools.groupby(column):
            if color != 0:
                length = len(list(group))
                if length > max_length:
                    max_length = length
                    max_color = color
    return max_color
