from rob_agi.colored_grid import ColoredGrid

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring the largest 'L'-shaped region
    to green (3). The algorithm follows these steps:
    1. Scan the grid from bottom to top, right to left.
    2. For each black (0) cell, calculate the largest possible 'L' shape.
    3. Keep track of the best 'L' shape (largest, and lowest/rightmost in case of ties).
    4. Color the best 'L' shape green (3).

    The 'L' shape is defined by its bottom-right corner (which must be black),
    extending upwards and to the left as far as possible, potentially including non-black cells.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    best_l = (0, (-1, -1), 0, 0)  # (size, (row, col), vertical_length, horizontal_length)

    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            if output_grid.get_cell(r, c) == 0:  # If it's a black cell
                vertical_length = 1
                while r - vertical_length >= 0:
                    vertical_length += 1
                vertical_length -= 1

                horizontal_length = 1
                while c - horizontal_length >= 0:
                    horizontal_length += 1
                horizontal_length -= 1

                l_size = vertical_length + horizontal_length - 1
                if l_size > best_l[0] or (l_size == best_l[0] and (r, c) > best_l[1]):
                    best_l = (l_size, (r, c), vertical_length, horizontal_length)

    if best_l[0] > 0:
        r, c = best_l[1]
        for i in range(best_l[2]):
            output_grid.set_cell(r - i, c, 3)
        for i in range(1, best_l[3]):  # Start from 1 to avoid double-coloring the corner
            output_grid.set_cell(r, c - i, 3)

    return output_grid
