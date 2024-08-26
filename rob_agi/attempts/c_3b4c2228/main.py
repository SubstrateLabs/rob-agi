from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_3b4c2228(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x3 output grid based on the presence and position of 2x2 squares.
    
    The transformation follows these rules:
    1. If any quadrant contains a 2x2 square of the same non-black color, set output[0][0] to blue (1).
    2. If (top-left and bottom-right) OR (top-right and bottom-left) quadrants contain 2x2 squares,
       AND these are the only quadrants with 2x2 squares, set output[1][1] to blue (1).
    3. If all quadrants contain 2x2 squares, set output[2][2] to blue (1).
    
    The input grid is divided into four quadrants, and the presence of 2x2 squares in these quadrants 
    determines the pattern in the output grid.
    """
    height, width = input_grid.get_dimensions()
    output_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def has_2x2_square(quadrant: Tuple[int, int, int, int]) -> bool:
        top, left, bottom, right = quadrant
        for r in range(top, bottom - 1):
            for c in range(left, right - 1):
                color = input_grid.values[r][c]
                if color != 0 and all(input_grid.values[r+dr][c+dc] == color 
                                      for dr, dc in [(0,1), (1,0), (1,1)]):
                    return True
        return False

    quadrants = [
        (0, 0, height//2, width//2),           # Top-left
        (0, width//2, height//2, width),       # Top-right
        (height//2, 0, height, width//2),      # Bottom-left
        (height//2, width//2, height, width)   # Bottom-right
    ]

    has_square = [has_2x2_square(quad) for quad in quadrants]

    if any(has_square):
        output_grid[0][0] = 1

    if (has_square[0] and has_square[3] and not has_square[1] and not has_square[2]) or \
       (has_square[1] and has_square[2] and not has_square[0] and not has_square[3]):
        output_grid[1][1] = 1

    if all(has_square):
        output_grid[2][2] = 1

    return ColoredGrid(values=output_grid)
