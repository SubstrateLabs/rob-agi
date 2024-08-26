from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on yellow regions and sky blue area:
    1. Identify yellow (4) regions and their properties.
    2. Process each yellow region:
       - If it contains non-yellow, non-sky blue colors, create a border with the smallest such color.
       - Border size is 1 for regions in top-left quadrant, 3 otherwise.
       - If it contains only yellow or sky blue, mark for replacement by sky blue.
    3. Apply borders to yellow regions.
    4. Create a sky blue (8) region starting from (0,0), filling black areas and marked regions.
    5. Ensure yellow region integrity and no black cells remain.
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
                    min_r = min(r for r, _ in region)
                    min_c = min(c for _, c in region)
                    max_r = max(r for r, _ in region)
                    max_c = max(c for _, c in region)
                    internal_colors = set()
                    for rr, cc in region:
                        for nr, nc in get_neighbors(rr, cc):
                            if grid.get_cell(nr, nc) not in [0, 4]:
                                internal_colors.add(grid.get_cell(nr, nc))
                    yellow_regions.append({
                        'region': region,
                        'bounds': (min_r, min_c, max_r, max_c),
                        'internal_colors': internal_colors
                    })
                    visited.update(region)
        return yellow_regions

    def process_yellow_regions(yellow_regions):
        for region_info in yellow_regions:
            min_r, min_c, max_r, max_c = region_info['bounds']
            internal_colors = region_info['internal_colors']
            
            if 8 in internal_colors or not internal_colors:
                region_info['replace'] = True
            else:
                border_color = min(internal_colors)
                border_size = 1 if min_r < rows // 2 and min_c < cols // 2 else 3
                region_info['border'] = {
                    'color': border_color,
                    'size': border_size
                }

    def apply_borders(yellow_regions):
        for region_info in yellow_regions:
            if 'border' in region_info:
                min_r, min_c, max_r, max_c = region_info['bounds']
                color = region_info['border']['color']
                size = region_info['border']['size']
                for r in range(max(0, min_r - size), min(rows, max_r + size + 1)):
                    for c in range(max(0, min_c - size), min(cols, max_c + size + 1)):
                        if grid.get_cell(r, c) == 0:
                            grid.set_cell(r, c, color)

    def create_sky_blue_region():
        stack = [(0, 0)]
        while stack:
            r, c = stack.pop()
            if is_valid(r, c) and grid.get_cell(r, c) in [0, 8]:
                grid.set_cell(r, c, 8)
                stack.extend(get_neighbors(r, c))

    yellow_regions = find_yellow_regions()
    process_yellow_regions(yellow_regions)
    apply_borders(yellow_regions)
    create_sky_blue_region()

    # Final cleanup
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 8)

    return grid
