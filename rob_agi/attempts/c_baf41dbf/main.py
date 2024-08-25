from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger rectangle,
    respecting the boundaries set by magenta (6) dots and grid edges.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Finds the maximum possible expansion in all directions.
    4. Creates a new grid with the expanded dimensions.
    5. Scales and recreates the shape in the new grid, preserving internal structure.
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

    # 3. Find maximum possible expansion
    def find_boundary(coord, step, limit):
        while 0 <= coord + step < limit:
            if any((coord + step == r and step != 0) or (coord + step == c and step == 0) for r, c in magenta_dots):
                break
            coord += step
        return coord

    left = find_boundary(min_c, -1, cols)
    right = find_boundary(max_c, 1, cols)
    top = find_boundary(min_r, -1, rows)
    bottom = find_boundary(max_r, 1, rows)

    # 4. Create new grid with expanded dimensions
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # 5. Scale and recreate the shape
    new_width = right - left + 1
    new_height = bottom - top + 1
    x_scale = new_width / (max_c - min_c + 1)
    y_scale = new_height / (max_r - min_r + 1)

    for new_r in range(top, bottom + 1):
        for new_c in range(left, right + 1):
            orig_r = min_r + (new_r - top) / y_scale
            orig_c = min_c + (new_c - left) / x_scale
            orig_r_int, orig_c_int = int(orig_r), int(orig_c)
            if (orig_r_int, orig_c_int) in green_cells:
                new_grid.set_cell(new_r, new_c, 3)
            elif input_grid.get_cell(orig_r_int, orig_c_int) == 0:
                new_grid.set_cell(new_r, new_c, 0)

    # 6. Add magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
