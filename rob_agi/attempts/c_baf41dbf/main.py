from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger rectangle,
    respecting the boundaries set by magenta (6) dots and grid edges.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Calculates the aspect ratio of the original shape.
    4. Expands the shape while maintaining the aspect ratio, stopping at grid edges or before magenta dots.
    5. Scales and preserves the interior structure of the original green shape.
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

    # 3. Calculate aspect ratio
    original_width = max_c - min_c + 1
    original_height = max_r - min_r + 1
    aspect_ratio = original_width / original_height

    # 4. Expand the shape
    def find_boundary(coord, step, limit):
        while coord + step >= 0 and coord + step < limit:
            if any((coord + step == r and step != 0) or (coord + step == c and step == 0) for r, c in magenta_dots):
                break
            coord += step
        return coord

    left = find_boundary(min_c, -1, cols)
    right = find_boundary(max_c, 1, cols)
    top = find_boundary(min_r, -1, rows)
    bottom = find_boundary(max_r, 1, rows)

    new_width = right - left + 1
    new_height = bottom - top + 1

    # Adjust to maintain aspect ratio
    if new_width / new_height > aspect_ratio:
        new_width = int(new_height * aspect_ratio)
        right = left + new_width - 1
    else:
        new_height = int(new_width / aspect_ratio)
        bottom = top + new_height - 1

    # 5. Create new grid and scale internal structure
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            new_grid.set_cell(r, c, 3)

    for r, c in green_cells:
        if input_grid.get_cell(r, c) == 0:
            new_r = int((r - min_r) * scale_y) + top
            new_c = int((c - min_c) * scale_x) + left
            new_grid.set_cell(new_r, new_c, 0)

    # 6. Add magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
