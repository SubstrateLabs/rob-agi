from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_fb791726(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Expanding its size to (2n) x (2n), where n is the input grid size
    2. Adding a single green (3) separator row and column in the middle
    3. Copying non-black cells to their new positions:
       - Top-left quadrant stays in place
       - Top-right quadrant moves to bottom-left
       - Bottom-left quadrant moves to top-right
       - Bottom-right quadrant stays in place
    4. Filling the rest with black (0)
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    # Add green separators
    mid_row, mid_col = input_rows, input_cols
    output_grid.values[mid_row - 1] = [3] * output_cols
    for i in range(output_rows):
        output_grid.values[i][mid_col - 1] = 3

    def map_coordinates(row: int, col: int, n: int) -> Tuple[int, int]:
        mid = n // 2
        if row < mid and col < mid:  # Top-left quadrant
            return row, col
        elif row < mid and col >= mid:  # Top-right quadrant
            return row + n, col
        elif row >= mid and col < mid:  # Bottom-left quadrant
            return row, col + n
        else:  # Bottom-right quadrant
            return row + n, col + n

    # Copy and transform the input grid
    for i in range(input_rows):
        for j in range(input_cols):
            if input_grid.values[i][j] != 0:
                new_i, new_j = map_coordinates(i, j, input_rows)
                output_grid.values[new_i][new_j] = input_grid.values[i][j]

    return output_grid
