from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Identify all black areas in the grid.
    2. Extend horizontal patterns into black areas from left and right.
    3. Extend vertical patterns into black areas from top and bottom.
    4. Resolve conflicts at intersections by preferring horizontal extensions.
    5. Fill any remaining isolated black cells with the most frequent surrounding color.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Helper function to get non-black color in a direction
    def get_non_black_color(r: int, c: int, dr: int, dc: int) -> int:
        while 0 <= r < rows and 0 <= c < cols:
            color = grid.get_cell(r, c)
            if color != 0:
                return color
            r += dr
            c += dc
        return -1  # Return -1 if no non-black color found

    # Extend horizontal patterns
    for r in range(rows):
        start = -1
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                if start == -1:
                    start = c
            elif start != -1:
                left_color = get_non_black_color(r, start - 1, 0, -1)
                right_color = get_non_black_color(r, c, 0, 1)
                fill_color = left_color if left_color != -1 else right_color
                for fill_c in range(start, c):
                    grid.set_cell(r, fill_c, fill_color)
                start = -1

    # Extend vertical patterns
    for c in range(cols):
        start = -1
        for r in range(rows):
            if grid.get_cell(r, c) == 0:
                if start == -1:
                    start = r
            elif start != -1:
                top_color = get_non_black_color(start - 1, c, -1, 0)
                bottom_color = get_non_black_color(r, c, 1, 0)
                fill_color = top_color if top_color != -1 else bottom_color
                for fill_r in range(start, r):
                    if grid.get_cell(fill_r, c) == 0:  # Only fill if still black
                        grid.set_cell(fill_r, c, fill_color)
                start = -1

    # Fill any remaining black cells with most frequent neighbor
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                        neighbors.append(grid.get_cell(nr, nc))
                if neighbors:
                    grid.set_cell(r, c, max(set(neighbors), key=neighbors.count))

    return grid
