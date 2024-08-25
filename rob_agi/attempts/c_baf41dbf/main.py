from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_baf41dbf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the green (3) shape into a larger rectangle,
    respecting the boundaries set by magenta (6) dots and grid edges.

    1. Analyzes the input grid to find green cells and magenta dots.
    2. Determines the bounding box of the original green shape.
    3. Finds the maximum possible expansion in all directions.
    4. Creates a new grid with the original dimensions.
    5. Fills the new rectangle with green, preserving the internal structure.
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

    # 4. Create new grid with original dimensions
    new_grid = input_grid.deep_copy()

    # 5. Fill the new rectangle and preserve internal structure
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if left <= c <= right and top <= r <= bottom:
                orig_r = min_r + (r - top) * (max_r - min_r) // (bottom - top)
                orig_c = min_c + (c - left) * (max_c - min_c) // (right - left)
                if (orig_r, orig_c) in green_cells:
                    new_grid.set_cell(r, c, 3)
                else:
                    new_grid.set_cell(r, c, 0)

    # 6. Restore magenta dots
    for r, c in magenta_dots:
        new_grid.set_cell(r, c, 6)

    return new_grid
