from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors vertically and horizontally based on the following rules:
    1. Identify all non-zero color points in the input grid.
    2. Create vertical lines for each color point, extending up and down until hitting another color or edge.
    3. Create horizontal lines for each color point, extending left and right until hitting another color or edge.
    4. Handle intersections by giving priority to lines from higher or more left starting points.
    5. Fill remaining black cells with the nearest non-black color above or to the left.
    6. Preserve original non-black cells in their positions.
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
            if new_grid.values[r][col] == 0:
                new_grid.values[r][col] = color

    # Process horizontal lines
    for row, col, color in color_points:
        # Extend left
        for c in range(col, -1, -1):
            if new_grid.values[row][c] != 0 and new_grid.values[row][c] != color:
                break
            new_grid.values[row][c] = color
        # Extend right
        for c in range(col, width):
            if new_grid.values[row][c] == 0 or new_grid.values[row][c] == color:
                new_grid.values[row][c] = color
            else:
                break

    # Handle color changes in columns
    for col in range(width):
        last_color = 0
        for row in range(height):
            if new_grid.values[row][col] != 0:
                last_color = new_grid.values[row][col]
            elif last_color != 0:
                new_grid.values[row][col] = last_color

    # Preserve original colors
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] != 0:
                new_grid.values[row][col] = input_grid.values[row][col]

    # Final check: fill remaining black cells
    for row in range(height):
        for col in range(width):
            if new_grid.values[row][col] == 0 and input_grid.values[row][col] == 0:
                # Find nearest non-black color above or to the left
                for r in range(row, -1, -1):
                    if new_grid.values[r][col] != 0:
                        new_grid.values[row][col] = new_grid.values[r][col]
                        break
                if new_grid.values[row][col] == 0:
                    for c in range(col, -1, -1):
                        if new_grid.values[row][c] != 0:
                            new_grid.values[row][col] = new_grid.values[row][c]
                            break

    return new_grid
