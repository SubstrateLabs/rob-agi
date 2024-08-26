from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, NamedTuple

class Rectangle(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int
    special_count: int
    area: int

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the blue rectangle with the most
    special squares (yellow or magenta), largest area, and most top-left position.

    The function performs the following steps:
    1. Scan the grid for special squares (yellow 4 or magenta 6).
    2. For each special square, expand to find the largest blue rectangle containing it.
    3. Evaluate each rectangle based on special square count, area, and position.
    4. Select the best rectangle based on these criteria.
    5. Construct and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and input_grid.get_cell(r, c) in [1, 4, 6]

    def expand_rectangle(start_r: int, start_c: int) -> Rectangle:
        top, left, bottom, right = start_r, start_c, start_r, start_c
        while top > 0 and all(is_valid_cell(top-1, c) for c in range(left, right+1)):
            top -= 1
        while bottom < rows-1 and all(is_valid_cell(bottom+1, c) for c in range(left, right+1)):
            bottom += 1
        while left > 0 and all(is_valid_cell(r, left-1) for r in range(top, bottom+1)):
            left -= 1
        while right < cols-1 and all(is_valid_cell(r, right+1) for r in range(top, bottom+1)):
            right += 1
        
        special_count = sum(1 for r in range(top, bottom+1) for c in range(left, right+1) 
                            if input_grid.get_cell(r, c) in [4, 6])
        area = (bottom - top + 1) * (right - left + 1)
        
        return Rectangle(top, left, bottom, right, special_count, area)

    # Generate candidate rectangles
    candidate_rectangles = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) in [4, 6]:
                rect = expand_rectangle(r, c)
                candidate_rectangles.append(rect)

    # Select the best rectangle
    if not candidate_rectangles:
        return ColoredGrid(values=[[]])  # Return an empty grid if no special squares found

    best_rect = max(candidate_rectangles, key=lambda x: (x.special_count, x.area, -x.top, -x.left))

    # Construct the output grid
    result = ColoredGrid(values=[
        [input_grid.get_cell(r, c) for c in range(best_rect.left, best_rect.right+1)]
        for r in range(best_rect.top, best_rect.bottom+1)
    ])

    return result
