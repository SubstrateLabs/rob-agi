from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger rectangle,
    respecting the boundaries set by magenta (6) dots and grid edges.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Calculates the maximum possible expansion in all directions.
    4. Creates a new grid with the original dimensions.
    5. Maps the original green shape to the new expanded area, preserving internal structure.
    6. Restores the original magenta dots.
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

    # 3. Calculate maximum possible expansion
    def find_boundary(coord, step, limit, axis):
        while 0 <= coord + step < limit:
            if any((coord + step == r and axis == 'row') or (coord + step == c and axis == 'col') for r, c in magenta_dots):
                break
            coord += step
        return coord

    left = find_boundary(min_c, -1, cols, 'col')
    right = find_boundary(max_c, 1, cols, 'col')
    top = find_boundary(min_r, -1, rows, 'row')
    bottom = find_boundary(max_r, 1, rows, 'row')

    # 4. Create new grid with original dimensions
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # 5. Map and fill the new expanded area
    orig_width = max_c - min_c + 1
    orig_height = max_r - min_r + 1
    new_width = right - left + 1
    new_height = bottom - top + 1

    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            orig_r = min_r + (r - top) * orig_height // new_height
            orig_c = min_c + (c - left) * orig_width // new_width
            if (orig_r, orig_c) in green_cells:
                new_grid.set_cell(r, c, 3)

    # 6. Restore magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
