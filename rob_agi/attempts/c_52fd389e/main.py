from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on yellow regions and sky blue area:
    1. Identify yellow (4) regions and their internal colors.
    2. Surround yellow regions with borders of their internal color or blue.
    3. Create a sky blue (8) region starting from 2 cells above and left of the rightmost, bottommost yellow region.
    4. Expand border colors to fill adjacent black cells.
    5. Clean up any remaining black cells.
    6. Ensure yellow region integrity.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(r+dr, c+dc)]

    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        region = []
        stack = [(r, c)]
        visited = set()
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                region.append((curr_r, curr_c))
                visited.add((curr_r, curr_c))
                stack.extend(get_neighbors(curr_r, curr_c))
        return region

    def find_yellow_regions():
        yellow_regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) == 4:
                    region = flood_fill(r, c, 4)
                    yellow_regions.append(region)
                    visited.update(region)
        return yellow_regions

    def find_internal_color(region):
        for r, c in region:
            for nr, nc in get_neighbors(r, c):
                if grid.get_cell(nr, nc) not in [0, 4]:
                    return grid.get_cell(nr, nc)
        return 1  # Default to blue if no internal color found

    def surround_region(region, color):
        for r, c in region:
            for nr, nc in get_neighbors(r, c):
                if grid.get_cell(nr, nc) == 0:
                    grid.set_cell(nr, nc, color)

    def find_sky_blue_start(yellow_regions):
        rightmost_bottom = max(
            (max((c for _, c in region), default=0), max((r for r, _ in region), default=0))
            for region in yellow_regions
        )
        return max(0, rightmost_bottom[1] - 2), max(0, rightmost_bottom[0] - 2)

    def create_sky_blue_region(start_col, start_row):
        for r in range(start_row, rows):
            for c in range(start_col, cols):
                if grid.get_cell(r, c) in [0, 8]:
                    grid.set_cell(r, c, 8)

    def expand_border_colors():
        for color in set(grid.get_cell(r, c) for r in range(rows) for c in range(cols)) - {0, 4, 8}:
            expansion = True
            while expansion:
                expansion = False
                for r in range(rows):
                    for c in range(cols):
                        if grid.get_cell(r, c) == color:
                            for nr, nc in get_neighbors(r, c):
                                if grid.get_cell(nr, nc) == 0:
                                    grid.set_cell(nr, nc, color)
                                    expansion = True

    def clean_up_black_cells():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:
                    neighbors = get_neighbors(r, c)
                    colors = [grid.get_cell(nr, nc) for nr, nc in neighbors if grid.get_cell(nr, nc) != 0]
                    if colors:
                        grid.set_cell(r, c, max(set(colors), key=colors.count))

    def verify_yellow_integrity():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 4:
                    if not any(grid.get_cell(nr, nc) == 4 for nr, nc in get_neighbors(r, c)):
                        surrounding_colors = [grid.get_cell(nr, nc) for nr, nc in get_neighbors(r, c)]
                        grid.set_cell(r, c, max(set(surrounding_colors), key=surrounding_colors.count))

    yellow_regions = find_yellow_regions()
    for region in yellow_regions:
        border_color = find_internal_color(region)
        surround_region(region, border_color)

    sky_start_col, sky_start_row = find_sky_blue_start(yellow_regions)
    create_sky_blue_region(sky_start_col, sky_start_row)

    expand_border_colors()
    clean_up_black_cells()
    verify_yellow_integrity()

    return grid
