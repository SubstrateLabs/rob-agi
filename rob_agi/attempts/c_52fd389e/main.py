from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque, defaultdict

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on yellow regions and their properties:
    1. Identify yellow (4) regions and analyze their properties.
    2. Process each yellow region:
       - If it contains sky blue (8), replace the entire region with sky blue.
       - Otherwise, create a border with the most frequent non-yellow, non-black color found.
       - Border size is 1 for small regions (<=4x4) in top-left quadrant, 
         2 for larger regions in top-left quadrant, and 3 for regions in other quadrants.
    3. Apply borders to yellow regions or replace with sky blue.
    4. Keep all remaining black (0) areas as they are.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    transformation_map = [[0 for _ in range(cols)] for _ in range(rows)]

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

    def get_quadrant(min_r: int, min_c: int) -> str:
        if min_r < rows // 2:
            return "top-left" if min_c < cols // 2 else "top-right"
        else:
            return "bottom-left" if min_c < cols // 2 else "bottom-right"

    def count_colors(region: Set[Tuple[int, int]]) -> Dict[int, int]:
        color_count = defaultdict(int)
        for r, c in region:
            for nr, nc in get_neighbors(r, c):
                cell_color = grid.get_cell(nr, nc)
                if cell_color not in [0, 4]:
                    color_count[cell_color] += 1
        return color_count

    def mark_border(region: Set[Tuple[int, int]], color: int, thickness: int):
        for r, c in region:
            for dr in range(-thickness, thickness + 1):
                for dc in range(-thickness, thickness + 1):
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc) and (nr, nc) not in region:
                        transformation_map[nr][nc] = color

    def find_yellow_regions():
        yellow_regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) == 4:
                    region = flood_fill(r, c, 4)
                    min_r, min_c = min(region)
                    max_r, max_c = max(region)
                    color_count = count_colors(region)
                    quadrant = get_quadrant(min_r, min_c)
                    yellow_regions.append({
                        'region': region,
                        'bounds': (min_r, min_c, max_r, max_c),
                        'color_count': color_count,
                        'quadrant': quadrant,
                        'size': (max_r - min_r + 1, max_c - min_c + 1)
                    })
                    visited.update(region)
        return yellow_regions

    def process_yellow_regions(yellow_regions):
        for region_info in yellow_regions:
            if 8 in region_info['color_count']:
                for r, c in region_info['region']:
                    transformation_map[r][c] = 8
            else:
                border_color = max(region_info['color_count'], key=region_info['color_count'].get) if region_info['color_count'] else 8
                if region_info['quadrant'] == "top-left":
                    border_size = 1 if max(region_info['size']) <= 4 else 2
                else:
                    border_size = 3
                mark_border(region_info['region'], border_color, border_size)

    def apply_transformations():
        for r in range(rows):
            for c in range(cols):
                if transformation_map[r][c] != 0:
                    grid.set_cell(r, c, transformation_map[r][c])

    yellow_regions = find_yellow_regions()
    process_yellow_regions(yellow_regions)
    apply_transformations()

    return grid
