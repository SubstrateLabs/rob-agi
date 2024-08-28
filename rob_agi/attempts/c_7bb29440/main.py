from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, NamedTuple

class Rectangle(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int
    area: int
    special_count: int

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the blue rectangle
    containing the most special squares (yellow or magenta), with largest area as a tiebreaker.

    The function performs the following steps:
    1. Scan the grid for blue cells (value 1).
    2. For each blue cell, expand to find the largest blue rectangle starting from that cell.
    3. Count the special squares (yellow 4 or magenta 6) within each rectangle.
    4. Select the rectangle with the most special squares.
    5. If multiple rectangles have the same number of special squares, choose the largest one.
    6. If multiple rectangles have the same special square count and size, choose the most top-left one.
    7. Construct and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()

    def is_blue_or_special(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and input_grid.get_cell(r, c) in [1, 4, 6]

    def expand_rectangle(start_r: int, start_c: int) -> Rectangle:
        right = start_c
        bottom = start_r
        while right + 1 < cols and is_blue_or_special(start_r, right + 1):
            right += 1
        while bottom + 1 < rows and all(is_blue_or_special(bottom + 1, c) for c in range(start_c, right + 1)):
            bottom += 1
        
        area = (bottom - start_r + 1) * (right - start_c + 1)
        special_count = sum(1 for r in range(start_r, bottom + 1) for c in range(start_c, right + 1) if input_grid.get_cell(r, c) in [4, 6])
        
        return Rectangle(start_r, start_c, bottom, right, area, special_count)

    # Generate candidate rectangles
    candidate_rectangles = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 1:  # Start from blue cells
                rect = expand_rectangle(r, c)
                if rect.special_count > 0:  # Only consider rectangles with special squares
                    candidate_rectangles.append(rect)

    # Select the best rectangle
    if not candidate_rectangles:
        return ColoredGrid(values=[[]])  # Return an empty grid if no valid rectangles found

    best_rect = max(candidate_rectangles, key=lambda x: (x.special_count, x.area, -x.top, -x.left))

    # Construct the output grid
    result = ColoredGrid(values=[
        [input_grid.get_cell(r, c) for c in range(best_rect.left, best_rect.right + 1)]
        for r in range(best_rect.top, best_rect.bottom + 1)
    ])

    return result
