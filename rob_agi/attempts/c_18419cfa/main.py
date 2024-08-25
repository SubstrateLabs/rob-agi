from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_18419cfa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 18419cfa challenge by expanding red (2) patterns within sky blue (8) regions.
    
    The function identifies connected sky blue regions, finds red pixels within them,
    and expands the red patterns into 3x3 structures. It handles various patterns including
    single pixels, L-shapes, crosses, and other complex shapes. The expansion creates
    filled 3x3 squares for single pixels and L-shapes, and 3x3 square rings (hollow centers)
    for crosses and more complex shapes. The expanded pattern is then repeated vertically
    to fill the sky blue region, maintaining vertical symmetry. The expansion is contained
    within the bounds of each sky blue region, and non-sky blue areas are preserved.
    """
    grid = input_grid.deep_copy()
    sky_blue_regions = find_connected_regions(grid, 8)
    
    for region in sky_blue_regions:
        expand_region(grid, region)
    
    return grid

def find_connected_regions(grid: ColoredGrid, color: int) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []

    def dfs(r: int, c: int) -> Set[Tuple[int, int]]:
        stack = [(r, c)]
        region = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and grid.get_cell(r, c) == color:
                visited.add((r, c))
                region.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) == color:
                regions.append(dfs(r, c))

    return regions

def expand_region(grid: ColoredGrid, region: Set[Tuple[int, int]]):
    min_r, max_r, min_c, max_c = get_region_bounds(region)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    if height < 3 or width < 3:
        for r, c in region:
            grid.set_cell(r, c, 2)
        return

    red_pixels = set((r, c) for r, c in region if grid.get_cell(r, c) == 2)
    template = create_expansion_template(red_pixels, min_r, min_c, max_r, max_c)
    
    template_height = len(template)
    repetitions = height // template_height
    extra_space = height % template_height
    
    start_r = min_r + extra_space // 2
    
    for i in range(repetitions):
        for r in range(template_height):
            for c in range(width):
                if template[r][c] == 2:
                    grid.set_cell(start_r + i * template_height + r, min_c + c, 2)

def get_region_bounds(region: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return min_r, max_r, min_c, max_c

def create_expansion_template(red_pixels: Set[Tuple[int, int]], min_r: int, min_c: int, max_r: int, max_c: int) -> List[List[int]]:
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    template = [[0 for _ in range(width)] for _ in range(height)]
    
    for r, c in red_pixels:
        r, c = r - min_r, c - min_c
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if dr == 0 and dc == 0 and len(red_pixels) > 1:
                        template[nr][nc] = 0  # Hollow center for complex shapes
                    else:
                        template[nr][nc] = 2
    
    return template
