from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on yellow regions and sky blue area:
    1. Identify yellow (4) regions.
    2. Process each yellow region:
       - If it contains a non-yellow, non-sky blue color, create a border with that color.
       - If it contains sky blue or no other colors, mark for integration with sky blue area.
    3. Expand bordered regions into adjacent black cells.
    4. Create a sky blue (8) region starting from near the top-left of the largest yellow region.
    5. Expand sky blue to fill remaining black cells and marked regions.
    6. Ensure yellow region integrity and no black cells remain.
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

    def process_yellow_region(region):
        internal_colors = set()
        for r, c in region:
            for nr, nc in get_neighbors(r, c):
                color = grid.get_cell(nr, nc)
                if color not in [0, 4]:
                    internal_colors.add(color)
        
        if 8 in internal_colors or not internal_colors:
            return None  # Mark for integration with sky blue
        else:
            return min(internal_colors)  # Choose the smallest non-zero, non-4 color

    def create_border(region, color):
        for r, c in region:
            for nr, nc in get_neighbors(r, c):
                if grid.get_cell(nr, nc) == 0:
                    grid.set_cell(nr, nc, color)

    def expand_color(color):
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

    def find_sky_blue_start(yellow_regions):
        largest_region = max(yellow_regions, key=len)
        min_r = min(r for r, _ in largest_region)
        min_c = min(c for _, c in largest_region)
        return max(0, min_r - 1), max(0, min_c - 1)

    def create_sky_blue_region(start_r, start_c):
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if is_valid(r, c) and grid.get_cell(r, c) in [0, 8]:
                grid.set_cell(r, c, 8)
                stack.extend(get_neighbors(r, c))

    yellow_regions = find_yellow_regions()
    for region in yellow_regions:
        border_color = process_yellow_region(region)
        if border_color:
            create_border(region, border_color)
            expand_color(border_color)

    sky_start_row, sky_start_col = find_sky_blue_start(yellow_regions)
    create_sky_blue_region(sky_start_row, sky_start_col)

    # Final cleanup
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 8)

    return grid
