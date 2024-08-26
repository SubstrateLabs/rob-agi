from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring the largest 'L'-shaped region
    of black (0) squares to green (3). The algorithm follows these steps:
    1. Find all potential 'L' shapes in the grid
    2. Select the largest 'L' shape
    3. Color the chosen 'L' shape green (3)

    The transformation aims to create a single, large green 'L' shape while maintaining
    the overall structure of the grid. The 'L' shape is defined by its bottom-right corner,
    extending upwards and to the left as far as possible within black cells.
    """
    output_grid = input_grid.deep_copy()

    def find_l_shapes(grid):
        rows, cols = grid.get_dimensions()
        l_shapes = []
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:  # If it's a black cell
                    vertical_length = 1
                    horizontal_length = 1
                    # Extend upwards
                    while r - vertical_length >= 0 and grid.get_cell(r - vertical_length, c) == 0:
                        vertical_length += 1
                    # Extend left
                    while c - horizontal_length >= 0 and grid.get_cell(r, c - horizontal_length) == 0:
                        horizontal_length += 1
                    l_size = vertical_length + horizontal_length - 1
                    l_shapes.append(((r, c), l_size, vertical_length, horizontal_length))
        return l_shapes

    def select_best_l_shape(l_shapes):
        return max(l_shapes, key=lambda x: (x[1], x[2], x[0][0], x[0][1]))

    def color_l_shape(grid, corner, vertical_length, horizontal_length):
        r, c = corner
        # Color vertically
        for i in range(vertical_length):
            grid.set_cell(r - i, c, 3)
        # Color horizontally
        for i in range(1, horizontal_length):  # Start from 1 to avoid double-coloring the corner
            grid.set_cell(r, c - i, 3)

    l_shapes = find_l_shapes(output_grid)
    if l_shapes:
        best_l = select_best_l_shape(l_shapes)
        color_l_shape(output_grid, best_l[0], best_l[2], best_l[3])

    return output_grid
