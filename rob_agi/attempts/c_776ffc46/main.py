from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue plus shapes (5 pixels) are changed to the target color (red or green).
    2. The target color (red or green) is determined by the color enclosed in a gray border.
    3. Other blue shapes and colors remain unchanged.
    4. Gray (5) acts as a border and is not considered part of any region.
    5. All transformations are applied simultaneously.

    The solution identifies the target color from gray-enclosed regions,
    detects blue plus shapes, and transforms them to the target color.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5

    def find_target_color():
        for i in range(rows):
            for j in range(cols):
                if input_grid.values[i][j] == GRAY:
                    enclosed_color = find_enclosed_color(i, j)
                    if enclosed_color in [RED, GREEN]:
                        return enclosed_color
        return RED  # Default to red if no enclosed color found

    def find_enclosed_color(start_x: int, start_y: int) -> int:
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x, y = start_x + dx, start_y + dy
            if 0 <= x < rows and 0 <= y < cols and input_grid.values[x][y] in [RED, GREEN]:
                return input_grid.values[x][y]
        return 0

    def is_plus_shape(x: int, y: int) -> bool:
        if input_grid.values[x][y] != BLUE:
            return False
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols and input_grid.values[nx][ny] == BLUE):
                return False
        return True

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    target_color = find_target_color()

    transformed_grid = [row[:] for row in input_grid.values]
    for i in range(rows):
        for j in range(cols):
            if is_plus_shape(i, j):
                for dx, dy in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                    transformed_grid[i+dx][j+dy] = target_color

    return ColoredGrid(values=transformed_grid)
