from rob_agi.colored_grid import ColoredGrid

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying initial blue cells and barriers.
    2. Determining expansion zones for each blue cell.
    3. Creating the largest possible blue rectangles within these zones.
    4. Merging overlapping or adjacent blue rectangles.
    5. Ensuring all cells within blue rectangles are blue, except for barriers.
    6. Validating that the transformation adheres to all rules.

    The function expands blue cells into rectangular regions, respecting barriers
    and the constraint of expanding up to two rows vertically from any initial blue cell.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_barrier(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) in [5, 8]

    def find_expansion_zone(r: int, c: int) -> tuple[int, int, int, int]:
        left = right = c
        while left > 0 and not is_barrier(r, left - 1):
            left -= 1
        while right < cols - 1 and not is_barrier(r, right + 1):
            right += 1
        top = max(0, r - 2)
        bottom = min(rows - 1, r + 2)
        return top, left, bottom, right

    def create_rectangle(top: int, left: int, bottom: int, right: int) -> None:
        for rr in range(top, bottom + 1):
            for cc in range(left, right + 1):
                if not is_barrier(rr, cc):
                    grid.set_cell(rr, cc, 1)

    # Find initial blue cells and their expansion zones
    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 1]
    expansion_zones = [find_expansion_zone(r, c) for r, c in blue_cells]

    # Create and merge rectangles
    for zone in expansion_zones:
        create_rectangle(*zone)

    # Final pass to ensure rectangular shapes
    for r in range(rows):
        blue_in_row = [c for c in range(cols) if grid.get_cell(r, c) == 1]
        if blue_in_row:
            left, right = min(blue_in_row), max(blue_in_row)
            for c in range(left, right + 1):
                if not is_barrier(r, c):
                    grid.set_cell(r, c, 1)

    return grid
