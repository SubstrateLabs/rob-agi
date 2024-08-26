from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_a04b2602(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying blue patterns to green regions.
    
    The transformation follows these steps:
    1. Identify contiguous green (3) regions.
    2. For each green region:
       a. Find red (2) dots within the region.
       b. Create blue (1) patterns around red dots, typically 3x3 squares.
       c. Expand and connect blue patterns within the green area.
       d. Preserve some green cells, especially near edges.
    3. Handle small green regions and edge cases.
    4. Preserve original red dots and black areas outside green regions.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] if is_valid(r+dr, c+dc)]

    def find_green_regions() -> List[List[Tuple[int, int]]]:
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and input_grid.get_cell(r, c) == 3:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and input_grid.get_cell(curr_r, curr_c) == 3:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            stack.extend(get_neighbors(curr_r, curr_c))
                    regions.append(region)
        return regions

    def create_blue_pattern(region: List[Tuple[int, int]]):
        red_dots = [(r, c) for r, c in region if input_grid.get_cell(r, c) == 2]
        for r, c in red_dots:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in region and (dr != 0 or dc != 0):
                        output_grid.set_cell(nr, nc, 1)  # Set to blue

    green_regions = find_green_regions()
    for region in green_regions:
        create_blue_pattern(region)

    return output_grid
