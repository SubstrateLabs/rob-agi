from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger rectangle,
    respecting the boundaries set by magenta (6) dots and grid edges.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Expands the bounding box in all directions, stopping at grid edges or before magenta dots.
    4. Creates a new rectangular shape based on the expanded bounding box.
    5. Preserves the interior structure of the original green shape.
    6. Adds the original magenta dots to the new grid.
    7. Returns the transformed grid.
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

    # 2. Determine the initial bounding box
    min_r = min(r for r, _ in green_cells)
    max_r = max(r for r, _ in green_cells)
    min_c = min(c for _, c in green_cells)
    max_c = max(c for _, c in green_cells)

    # 3. Expand the bounding box
    def find_boundary(coords, compare_func, limit, step):
        return next((i for i in range(compare_func(coords), limit, step)
                     if any(coord == i for coord in (r if step != 0 else c for r, c in magenta_dots))),
                    limit)

    left = find_boundary(min_c, lambda x: x - 1, 0, -1)
    right = find_boundary(max_c, lambda x: x + 1, cols - 1, 1)
    top = find_boundary(min_r, lambda x: x - 1, 0, -1)
    bottom = find_boundary(max_r, lambda x: x + 1, rows - 1, 1)

    # 4. Create the new rectangular shape
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            new_grid.set_cell(r, c, 3)

    # 5. Preserve the interior structure
    for r, c in green_cells:
        if input_grid.get_cell(r, c) == 0:
            new_grid.set_cell(r, c, 0)

    # 6. Add magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
