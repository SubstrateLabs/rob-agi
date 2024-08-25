from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_0692e18c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 pattern.
    The expansion pattern depends on the color of the cell:
    - Orange (7): Expands into a 3x3 cross shape
    - Magenta (6): Expands into a 3x3 L-shape in the top-left corner
    - Yellow (4): Expands into a 3x3 square with top and left sides filled
    - Black (0): Expands into a 3x3 empty space
    - Other colors: Expands into a 3x3 square with all sides filled and an empty center
    
    Each cell in the input grid is replaced by its corresponding 3x3 expansion pattern
    in the output grid, creating a 9x9 result that preserves the original structure
    while adding detail based on the color-specific patterns.
    """
    def expand_cell(color: int) -> List[List[int]]:
        if color == 7:  # orange
            return [[0, color, 0], [color, 0, color], [0, color, 0]]
        elif color == 6:  # magenta
            return [[color, color, 0], [color, 0, 0], [0, 0, 0]]
        elif color == 4:  # yellow
            return [[color, color, 0], [color, color, 0], [0, 0, 0]]
        elif color == 0:  # black
            return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        else:  # other colors
            return [[color, color, color], [color, 0, color], [color, color, color]]

    output_values = [[0 for _ in range(9)] for _ in range(9)]

    for row in range(3):
        for col in range(3):
            color = input_grid.values[row][col]
            if color != 0:
                pattern = expand_cell(color)
                top, left = row * 3, col * 3
                for i in range(3):
                    for j in range(3):
                        output_values[top + i][left + j] = pattern[i][j]

    return ColoredGrid(values=output_values)
