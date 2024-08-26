from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors vertically based on the following rules:
    1. Identify all non-zero color points in the input grid.
    2. Create vertical lines for each color point, extending up and down until hitting another color or edge.
    3. Handle intersections by giving priority to lines from higher starting points.
    4. Preserve original non-black cells in their positions.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    # Identify color points
    color_points = [(row, col, input_grid.values[row][col]) 
                    for row in range(height) 
                    for col in range(width) 
                    if input_grid.values[row][col] != 0]
    color_points.sort(key=lambda x: (x[0], x[1]))  # Sort top-to-bottom, then left-to-right

    # Process vertical lines
    for row, col, color in color_points:
        # Extend upwards
        for r in range(row, -1, -1):
            if new_grid.values[r][col] != 0:
                break
            new_grid.values[r][col] = color
        # Extend downwards
        for r in range(row, height):
            if new_grid.values[r][col] != 0 and new_grid.values[r][col] != color:
                break
            new_grid.values[r][col] = color

    # Preserve original colors
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] != 0:
                new_grid.values[row][col] = input_grid.values[row][col]

    return new_grid
