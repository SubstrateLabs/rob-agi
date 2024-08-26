from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the largest blue rectangle
    containing exactly one special square (yellow or magenta) and having
    the most top-left position.

    The function performs the following steps:
    1. Identify all yellow (4) and magenta (6) squares in the input grid.
    2. For each special square, expand to find the largest blue rectangle containing only that special square.
    3. Select the best rectangle based on size and position criteria.
    4. Construct and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()
    special_squares = []

    # Step 1: Identify special squares
    for r in range(rows):
        for c in range(cols):
            cell = input_grid.get_cell(r, c)
            if cell in [4, 6]:
                special_squares.append((r, c, cell))

    def expand_rectangle(r, c):
        top, left, bottom, right = r, c, r, c
        while top > 0 and all(input_grid.get_cell(top-1, col) == 1 for col in range(left, right+1)):
            top -= 1
        while bottom < rows-1 and all(input_grid.get_cell(bottom+1, col) == 1 for col in range(left, right+1)):
            bottom += 1
        while left > 0 and all(input_grid.get_cell(row, left-1) == 1 for row in range(top, bottom+1)):
            left -= 1
        while right < cols-1 and all(input_grid.get_cell(row, right+1) == 1 for row in range(top, bottom+1)):
            right += 1
        return (top, left, bottom, right)

    # Step 2: Generate and expand candidate rectangles
    candidate_rectangles = []
    for r, c, special_value in special_squares:
        rect = expand_rectangle(r, c)
        area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
        candidate_rectangles.append((*rect, area, special_value, r, c))

    # Step 3: Select the best rectangle
    if not candidate_rectangles:
        return None

    best_rect = max(candidate_rectangles, key=lambda x: (x[4], -x[0], -x[1]))

    # Step 4: Construct the output grid
    top, left, bottom, right, _, special_value, special_r, special_c = best_rect
    height = bottom - top + 1
    width = right - left + 1
    result = ColoredGrid(values=[[1 for _ in range(width)] for _ in range(height)])

    # Place the special square in its relative position
    result.set_cell(special_r - top, special_c - left, special_value)

    return result
