from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying initial blue cells and barriers.
    2. Determining expansion zones for each blue cell, respecting horizontal and vertical constraints.
    3. Merging overlapping or adjacent expansion zones.
    4. Creating blue rectangles within these zones, respecting barriers.
    5. Ensuring all cells within blue rectangles are blue, except for barriers.
    6. Performing a final pass to guarantee rectangular shapes.

    The function expands blue cells into rectangular regions, respecting barriers (gray and sky blue cells),
    and the constraints of expanding up to one cell horizontally and up to two rows vertically from any initial blue cell.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_barrier(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) in [5, 8]

    def find_expansion_zone(r: int, c: int) -> Tuple[int, int, int, int]:
        left = max(0, c - 1)
        right = min(cols - 1, c + 1)
        top = max(0, r - 2)
        bottom = min(rows - 1, r + 2)
        return top, left, bottom, right

    def merge_zones(zones: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        zones.sort()
        merged = []
        for zone in zones:
            if not merged or zone[0] > merged[-1][2] + 1 or zone[1] > merged[-1][3] + 1:
                merged.append(zone)
            else:
                merged[-1] = (
                    min(merged[-1][0], zone[0]),
                    min(merged[-1][1], zone[1]),
                    max(merged[-1][2], zone[2]),
                    max(merged[-1][3], zone[3])
                )
        return merged

    def create_rectangle(top: int, left: int, bottom: int, right: int) -> None:
        for rr in range(top, bottom + 1):
            for cc in range(left, right + 1):
                if not is_barrier(rr, cc):
                    grid.set_cell(rr, cc, 1)

    # Find initial blue cells and their expansion zones
    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 1]
    expansion_zones = [find_expansion_zone(r, c) for r, c in blue_cells]

    # Merge overlapping zones
    merged_zones = merge_zones(expansion_zones)

    # Create rectangles for merged zones
    for zone in merged_zones:
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
