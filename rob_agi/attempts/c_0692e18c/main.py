from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_0692e18c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 pattern.
    The expansion pattern depends on the color of the cell:
    - Orange (7): Expands into a 3x3 cross shape
    - Magenta (6): Expands into a 3x3 L-shape
    - Yellow (4): Expands into a 3x3 square with top and left sides filled
    - Black (0): Expands into a 3x3 empty space
    - Other colors: Expands into a 3x3 square with all sides filled and an empty center
    
    The orientation of the expanded pattern is determined by its position in the input grid,
    with all patterns pointing towards the center of the original grid.
    """
    def rotate_90_clockwise(pattern):
        return [list(row) for row in zip(*pattern[::-1])]

    def rotate_90_counterclockwise(pattern):
        return [list(row) for row in zip(*pattern)][::-1]

    def rotate_180(pattern):
        return [row[::-1] for row in pattern[::-1]]

    def flip_horizontal(pattern):
        return [row[::-1] for row in pattern]

    def flip_vertical(pattern):
        return pattern[::-1]

    def get_base_pattern(color: int) -> List[List[int]]:
        if color == 7:  # orange
            return [[0, color, 0], [color, 0, color], [0, color, 0]]
        elif color == 6:  # magenta
            return [[color, color, 0], [color, 0, 0], [0, 0, 0]]
        elif color == 4:  # yellow
            return [[color, color, 0], [color, 0, 0], [0, 0, 0]]
        elif color == 0:  # black
            return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        else:  # other colors
            return [[color, color, color], [color, 0, color], [color, color, color]]

    def orient_pattern(pattern: List[List[int]], row: int, col: int) -> List[List[int]]:
        if (row, col) == (0, 0):
            return rotate_180(pattern)
        elif (row, col) == (0, 1):
            return rotate_90_clockwise(pattern)
        elif (row, col) == (0, 2):
            return pattern
        elif (row, col) == (1, 0):
            return rotate_90_clockwise(pattern)
        elif (row, col) == (1, 1):
            return pattern
        elif (row, col) == (1, 2):
            return rotate_90_counterclockwise(pattern)
        elif (row, col) == (2, 0):
            return rotate_180(pattern)
        elif (row, col) == (2, 1):
            return rotate_90_counterclockwise(pattern)
        else:  # (2, 2)
            return pattern

    output_values = [[0 for _ in range(9)] for _ in range(9)]

    for row in range(3):
        for col in range(3):
            color = input_grid.values[row][col]
            pattern = get_base_pattern(color)
            oriented_pattern = orient_pattern(pattern, row, col)
            top, left = row * 3, col * 3
            for i in range(3):
                for j in range(3):
                    output_values[top + i][left + j] = oriented_pattern[i][j]

    return ColoredGrid(values=output_values)
