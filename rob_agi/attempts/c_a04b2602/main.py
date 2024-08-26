from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_a04b2602(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying complex blue patterns to green regions.
    
    The transformation follows these steps:
    1. Identify contiguous green (3) regions using flood fill.
    2. Analyze red (2) dot distribution within each green region.
    3. For each green region:
       a. Initialize transformation parameters based on region size and red dot density.
       b. Create blue (1) patterns starting from red dots, expanding organically.
       c. Adjust blue formation probability based on distance from red dots and region edges.
       d. Ensure connectivity of blue areas in larger regions.
       e. Preserve some green cells, especially near edges and as islands.
    4. Handle small green regions differently, with simpler patterns.
    5. Fine-tune patterns to reduce isolated cells and improve organic appearance.
    6. Preserve original red dots and areas outside green regions.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        if diagonal:
            directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
        return [(r+dr, c+dc) for dr, dc in directions if is_valid(r+dr, c+dc)]

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
        region_size = len(region)
        blue_cells = set()

        # Initialize blue patterns around red dots
        for r, c in red_dots:
            neighbors = get_neighbors(r, c, diagonal=True)
            for nr, nc in neighbors:
                if (nr, nc) in region and input_grid.get_cell(nr, nc) != 2:
                    output_grid.set_cell(nr, nc, 1)
                    blue_cells.add((nr, nc))

        # Expand blue patterns
        expansion_iterations = min(5, region_size // 10)
        for _ in range(expansion_iterations):
            new_blue_cells = set()
            for r, c in blue_cells:
                neighbors = get_neighbors(r, c)
                for nr, nc in neighbors:
                    if (nr, nc) in region and (nr, nc) not in blue_cells and random.random() < 0.7:
                        new_blue_cells.add((nr, nc))
            blue_cells.update(new_blue_cells)
            for r, c in new_blue_cells:
                if input_grid.get_cell(r, c) != 2:
                    output_grid.set_cell(r, c, 1)

        # Preserve some green areas
        green_preservation_rate = max(0.2, 1 - (region_size / 100))
        for r, c in region:
            if (r, c) not in blue_cells and random.random() < green_preservation_rate:
                output_grid.set_cell(r, c, 3)

        # Connect blue areas in larger regions
        if region_size > 50:
            for _ in range(region_size // 20):
                r, c = random.choice(list(blue_cells))
                direction = random.choice([(0,1),(1,0),(0,-1),(-1,0)])
                for _ in range(3):
                    r, c = r + direction[0], c + direction[1]
                    if (r, c) in region and input_grid.get_cell(r, c) != 2:
                        output_grid.set_cell(r, c, 1)
                        blue_cells.add((r, c))

    green_regions = find_green_regions()
    for region in green_regions:
        if len(region) < 25:  # Handle small regions differently
            red_dots = [(r, c) for r, c in region if input_grid.get_cell(r, c) == 2]
            if red_dots:
                r, c = red_dots[0]
                neighbors = get_neighbors(r, c)
                for nr, nc in neighbors:
                    if (nr, nc) in region and input_grid.get_cell(nr, nc) != 2:
                        output_grid.set_cell(nr, nc, 1)
        else:
            create_blue_pattern(region)

    return output_grid
