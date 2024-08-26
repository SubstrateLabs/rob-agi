from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on yellow regions and sky blue area:
    1. Identify yellow (4) regions and their properties.
    2. Process each yellow region:
       - If it contains sky blue (8), mark for replacement.
       - Otherwise, create a border with the smallest non-yellow color found.
       - Border size is 1 for small regions (<=4x4) in top-left quadrant, 3 otherwise.
    3. Apply borders to yellow regions or replace with sky blue.
    4. Fill all remaining black (0) areas with sky blue (8).
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(r+dr, c+dc)]

    def flood_fill(r: int, c: int, color: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(r, c)])
        while queue:
            curr_r, curr_c = queue.popleft()
            if (curr_r, curr_c) not in region and grid.get_cell(curr_r, curr_c) == color:
                region.add((curr_r, curr_c))
                queue.extend(get_neighbors(curr_r, curr_c))
        return region

    def find_yellow_regions():
        yellow_regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) == 4:
                    region = flood_fill(r, c, 4)
                    min_r, min_c = min(region)
                    max_r, max_c = max(region)
                    internal_colors = set()
                    for rr, cc in region:
                        for nr, nc in get_neighbors(rr, cc):
                            if grid.get_cell(nr, nc) not in [0, 4]:
                                internal_colors.add(grid.get_cell(nr, nc))
                    yellow_regions.append({
                        'region': region,
                        'bounds': (min_r, min_c, max_r, max_c),
                        'internal_colors': internal_colors,
                        'size': len(region)
                    })
                    visited.update(region)
        return yellow_regions

    def process_yellow_regions(yellow_regions):
        for region_info in yellow_regions:
            min_r, min_c, max_r, max_c = region_info['bounds']
            internal_colors = region_info['internal_colors']
            
            if 8 in internal_colors:
                region_info['replace'] = True
            else:
                border_color = min(internal_colors) if internal_colors else 8
                is_small_top_left = (max_r - min_r <= 4 and max_c - min_c <= 4 and
                                     min_r < rows // 2 and min_c < cols // 2)
                border_size = 1 if is_small_top_left else 3
                region_info['border'] = {
                    'color': border_color,
                    'size': border_size
                }

    def apply_transformations(yellow_regions):
        for region_info in yellow_regions:
            if region_info.get('replace', False):
                for r, c in region_info['region']:
                    grid.set_cell(r, c, 8)
            elif 'border' in region_info:
                min_r, min_c, max_r, max_c = region_info['bounds']
                color = region_info['border']['color']
                size = region_info['border']['size']
                for r in range(max(0, min_r - size), min(rows, max_r + size + 1)):
                    for c in range(max(0, min_c - size), min(cols, max_c + size + 1)):
                        if grid.get_cell(r, c) == 0:
                            grid.set_cell(r, c, color)

    def fill_remaining_space():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 8)

    yellow_regions = find_yellow_regions()
    process_yellow_regions(yellow_regions)
    apply_transformations(yellow_regions)
    fill_remaining_space()

    return grid
