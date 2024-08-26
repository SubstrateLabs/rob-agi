from rob_agi.colored_grid import ColoredGrid

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Scanning the grid to identify rows with initial blue cells (1) and barriers (5 or 8).
    2. Filling entire rows containing initial blue cells with blue (1), except for barriers.
    3. Expanding blue vertically up to 2 cells from filled rows, stopping at barriers or edges.
    4. Ensuring blue regions form rectangular shapes.
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

    # Step 1 & 2: Scan and fill rows with initial blue cells
    blue_rows = set()
    for r in range(rows):
        if 1 in [grid.get_cell(r, c) for c in range(cols)]:
            fill_row(r)
            blue_rows.add(r)

    # Step 3: Vertical expansion
    for r in blue_rows:
        vertical_expand(r)

    # Step 4: Ensure rectangular shapes
    for r in range(rows):
        blue_in_row = any(grid.get_cell(r, c) == 1 for c in range(cols))
        if blue_in_row:
            left = next(c for c in range(cols) if grid.get_cell(r, c) == 1)
            right = next(c for c in range(cols-1, -1, -1) if grid.get_cell(r, c) == 1)
            for c in range(left, right + 1):
                if not is_barrier(r, c):
                    grid.set_cell(r, c, 1)

    # Step 5: Final pass
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) not in [5, 8]:
                blue_neighbors = sum(1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                                     if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 1)
                if blue_neighbors > 0:
                    grid.set_cell(r, c, 1)

    return grid
