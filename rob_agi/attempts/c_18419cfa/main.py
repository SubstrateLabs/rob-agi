from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_18419cfa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 18419cfa challenge by expanding red (2) patterns within sky blue (8) regions.
    
    The function identifies connected sky blue regions, analyzes red patterns within them,
    and expands these patterns based on their shape and available space. It creates a
    symmetrical template from the red pattern, which is then repeated vertically to fill
    the sky blue region. The expansion maintains symmetry and is contained within the
    bounds of each sky blue region. Non-sky blue areas are preserved.
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
    
    red_pixels = set((r, c) for r, c in region if grid.get_cell(r, c) == 2)
    if not red_pixels:
        return

    template = create_symmetrical_template(red_pixels, min_r, min_c, max_r, max_c)
    
    template_height = len(template)
    repetitions = height // template_height
    extra_space = height % template_height
    
    start_r = min_r + extra_space // 2
    
    for i in range(repetitions):
        for r in range(template_height):
            for c in range(width):
                if template[r][c] == 2 and (start_r + i * template_height + r, min_c + c) in region:
                    grid.set_cell(start_r + i * template_height + r, min_c + c, 2)

    # Handle partial repetition at the bottom
    remaining_rows = height - (repetitions * template_height)
    if remaining_rows > 0:
        for r in range(min(remaining_rows, template_height)):
            for c in range(width):
                if template[r][c] == 2 and (start_r + repetitions * template_height + r, min_c + c) in region:
                    grid.set_cell(start_r + repetitions * template_height + r, min_c + c, 2)

def get_region_bounds(region: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return min_r, max_r, min_c, max_c

def create_symmetrical_template(red_pixels: Set[Tuple[int, int]], min_r: int, min_c: int, max_r: int, max_c: int) -> List[List[int]]:
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    template = [[0 for _ in range(width)] for _ in range(height * 2)]
    
    # Create the top half of the template
    for r, c in red_pixels:
        r, c = r - min_r, c - min_c
        template[r][c] = 2
    
    # Mirror the top half to create the bottom half
    for r in range(height):
        template[height * 2 - 1 - r] = template[r].copy()
    
    # If the template height is odd, add an extra row in the middle
    if height % 2 == 1:
        middle_row = [2 if any(template[height-1][c] == 2 or template[height][c] == 2 else 0 for c in range(width)]
        template.insert(height, middle_row)
    
    return template
