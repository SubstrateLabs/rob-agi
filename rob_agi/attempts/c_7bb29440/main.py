from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the largest blue rectangle
    containing at least one special square (yellow or magenta) and having
    the most top-left position.

    The function performs the following steps:
    1. Identify all yellow (4) and magenta (6) squares in the input grid.
    2. For each special square, expand to find the largest blue rectangle containing it.
    3. Select the best rectangle based on size and position criteria.
    4. Construct and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()

    def is_valid_cell(r, c):
        return 0 <= r < rows and 0 <= c < cols and input_grid.get_cell(r, c) in [1, 4, 6]

    def expand_rectangle(start_r, start_c):
        top, left, bottom, right = start_r, start_c, start_r, start_c
        while top > 0 and all(is_valid_cell(top-1, c) for c in range(left, right+1)):
            top -= 1
        while bottom < rows-1 and all(is_valid_cell(bottom+1, c) for c in range(left, right+1)):
            bottom += 1
        while left > 0 and all(is_valid_cell(r, left-1) for r in range(top, bottom+1)):
            left -= 1
        while right < cols-1 and all(is_valid_cell(r, right+1) for r in range(top, bottom+1)):
            right += 1
        return (top, left, bottom, right)

    # Generate candidate rectangles
    candidate_rectangles = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) in [4, 6]:
                rect = expand_rectangle(r, c)
                area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
                candidate_rectangles.append((*rect, area))

    # Select the best rectangle
    if not candidate_rectangles:
        return None

    best_rect = max(candidate_rectangles, key=lambda x: (x[4], -x[0], -x[1]))

    # Construct the output grid
    top, left, bottom, right, _ = best_rect
    height = bottom - top + 1
    width = right - left + 1
    result = ColoredGrid(values=[
        [input_grid.get_cell(r, c) for c in range(left, right+1)]
        for r in range(top, bottom+1)
    ])

    return result
