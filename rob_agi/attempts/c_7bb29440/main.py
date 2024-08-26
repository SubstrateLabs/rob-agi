from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the largest blue rectangle
    containing at least one special square (yellow or magenta) and having
    the most top-left position.

    The function performs the following steps:
    1. Identify all yellow (4) and magenta (6) squares in the input grid.
    2. Generate candidate rectangles based on special squares.
    3. Expand candidate rectangles to include as many blue (1) squares as possible.
    4. Select the best rectangle based on size and position criteria.
    5. Construct and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_squares = []
    magenta_squares = []

    # Step 1: Identify special squares
    for r in range(rows):
        for c in range(cols):
            cell = input_grid.get_cell(r, c)
            if cell == 4:
                yellow_squares.append((r, c))
            elif cell == 6:
                magenta_squares.append((r, c))

    def expand_rectangle(top, left, bottom, right):
        while top > 0 and all(input_grid.get_cell(top-1, c) in [1, 4, 6] for c in range(left, right+1)):
            top -= 1
        while bottom < rows-1 and all(input_grid.get_cell(bottom+1, c) in [1, 4, 6] for c in range(left, right+1)):
            bottom += 1
        while left > 0 and all(input_grid.get_cell(r, left-1) in [1, 4, 6] for r in range(top, bottom+1)):
            left -= 1
        while right < cols-1 and all(input_grid.get_cell(r, right+1) in [1, 4, 6] for r in range(top, bottom+1)):
            right += 1
        return (top, left, bottom, right)

    # Step 2 and 3: Generate and expand candidate rectangles
    candidate_rectangles = []
    special_squares = yellow_squares + magenta_squares
    for i, sq1 in enumerate(special_squares):
        for sq2 in special_squares[i:]:
            top = min(sq1[0], sq2[0])
            left = min(sq1[1], sq2[1])
            bottom = max(sq1[0], sq2[0])
            right = max(sq1[1], sq2[1])
            expanded = expand_rectangle(top, left, bottom, right)
            candidate_rectangles.append(expanded)

    # Step 4: Select the best rectangle
    if not candidate_rectangles:
        return None

    best_rect = max(candidate_rectangles, key=lambda r: ((r[2]-r[0]+1)*(r[3]-r[1]+1), -r[0], -r[1]))

    # Step 5: Construct the output grid
    height = best_rect[2] - best_rect[0] + 1
    width = best_rect[3] - best_rect[1] + 1
    result = ColoredGrid(values=[[1 for _ in range(width)] for _ in range(height)])

    for r in range(height):
        for c in range(width):
            cell = input_grid.get_cell(best_rect[0] + r, best_rect[1] + c)
            if cell in [4, 6]:
                result.set_cell(r, c, cell)

    return result
