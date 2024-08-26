from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Scanning the grid to identify rows with initial blue cells (1) and barriers (5 or 8).
    2. Filling entire rows containing initial blue cells with blue (1), except for barriers.
    3. Expanding blue vertically up to 2 cells from filled rows, stopping at barriers or edges.
    4. Creating a "blue aura" around barriers, extending up to 2 cells horizontally and vertically.
    5. Performing a final pass to ensure all marked cells are blue (1) and others unchanged.

    The function creates a specific pattern of blue expansion based on initial blue cells and barriers,
    following the "2 cells away" rule and respecting grid boundaries.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_barrier(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) in [5, 8]

    def fill_row(r: int):
        for c in range(cols):
            if not is_barrier(r, c):
                grid.set_cell(r, c, 1)

    def vertical_expand(r: int):
        for dr in [-2, -1, 1, 2]:
            nr = r + dr
            if 0 <= nr < rows:
                for c in range(cols):
                    if not is_barrier(nr, c) and grid.get_cell(nr, c) == 0:
                        grid.set_cell(nr, c, 1)

    def barrier_aura(r: int, c: int):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                    grid.set_cell(nr, nc, 1)

    # Step 1 & 2: Scan and fill rows with initial blue cells
    for r in range(rows):
        if 1 in [grid.get_cell(r, c) for c in range(cols)]:
            fill_row(r)

    # Step 3: Vertical expansion
    for r in range(rows):
        if 1 in [grid.get_cell(r, c) for c in range(cols)]:
            vertical_expand(r)

    # Step 4: Barrier aura
    for r in range(rows):
        for c in range(cols):
            if is_barrier(r, c):
                barrier_aura(r, c)

    return grid
