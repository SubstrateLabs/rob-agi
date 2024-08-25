from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger 'C' shape,
    respecting the boundaries set by magenta (6) dots.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Extends the bounding box in all directions, stopping at grid edges or before magenta dots.
    4. Creates a new 'C' shape based on the extended bounding box.
    5. Adds the original magenta dots to the new grid.
    6. Returns the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    green_cells = []
    magenta_dots = []

    # 1. Analyze the input grid
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 3:
                green_cells.append((r, c))
            elif input_grid.get_cell(r, c) == 6:
                magenta_dots.append((r, c))

    # 2. Determine the bounding box
    min_r = min(r for r, _ in green_cells)
    max_r = max(r for r, _ in green_cells)
    min_c = min(c for _, c in green_cells)
    max_c = max(c for _, c in green_cells)

    # 3. Extend the bounding box
    left = max(0, min(min_c, min(c for r, c in magenta_dots if c < min_c) + 1))
    right = min(cols - 1, max(max_c, max(c for r, c in magenta_dots if c > max_c) - 1))
    top = max(0, min(min_r, min(r for r, c in magenta_dots if r < min_r) + 1))
    bottom = min(rows - 1, max(max_r, max(r for r, c in magenta_dots if r > max_r) - 1))

    # 4. Create the new 'C' shape
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if c < right or min_c <= c <= max_c:
                new_grid.set_cell(r, c, 3)

    # 5. Add magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
