from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_0692e18c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 pattern.
    The expansion pattern depends on the color of the cell:
    - Orange (7): Expands into a 3x3 cross shape
    - Magenta (6): Expands into a 3x3 L-shape
    - Yellow (4): Expands into a 3x3 square with two opposite sides filled
    - Other colors: Expands into a 3x3 square with all sides filled
    - Black (0): Remains as empty space
    """
    def expand_cell(color: int) -> List[List[int]]:
        pattern = [
            [color, 0, color],
            [0, 0, 0],
            [color, 0, color]
        ]
        if color == 7:  # orange
            pattern[0][1] = pattern[1][0] = pattern[1][2] = pattern[2][1] = color
        elif color == 6:  # magenta
            pattern[0][1] = pattern[1][2] = color
        elif color == 4:  # yellow
            pattern[0][1] = pattern[2][1] = color
        elif color != 0:  # for any other non-black color, fill all sides
            pattern[0][1] = pattern[1][0] = pattern[1][2] = pattern[2][1] = color
        
        return pattern

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
