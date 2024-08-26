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
        red_dot_density = len(red_dots) / region_size
        blue_cells = set()
        queue = []

        # Initialize parameters
        base_probability = 0.7
        influence_radius = max(3, region_size // 20)
        edge_preservation_factor = 0.5

        # Initialize blue patterns around red dots
        for r, c in red_dots:
            neighbors = get_neighbors(r, c, diagonal=True)
            for nr, nc in neighbors:
                if (nr, nc) in region and input_grid.get_cell(nr, nc) != 2:
                    output_grid.set_cell(nr, nc, 1)
                    blue_cells.add((nr, nc))
                    queue.append((nr, nc))

        # Expand blue patterns
        while queue:
            r, c = queue.pop(0)
            neighbors = get_neighbors(r, c)
            for nr, nc in neighbors:
                if (nr, nc) in region and (nr, nc) not in blue_cells and input_grid.get_cell(nr, nc) != 2:
                    # Calculate distances
                    edge_distance = min(nr, nc, rows-1-nr, cols-1-nc)
                    red_dot_distance = min(abs(nr-rr) + abs(nc-cc) for rr, cc in red_dots)
                    
                    # Adjust probability
                    prob = base_probability
                    prob *= (1 - (red_dot_distance / influence_radius))
                    prob *= (1 + (edge_distance / influence_radius) * edge_preservation_factor)
                    prob *= (1 + red_dot_density)
                    
                    if random.random() < prob:
                        output_grid.set_cell(nr, nc, 1)
                        blue_cells.add((nr, nc))
                        queue.append((nr, nc))

        # Ensure connectivity
        blue_cells_list = list(blue_cells)
        if blue_cells_list:
            connected = set()
            stack = [blue_cells_list[0]]
            while stack:
                r, c = stack.pop()
                if (r, c) not in connected:
                    connected.add((r, c))
                    stack.extend([(nr, nc) for nr, nc in get_neighbors(r, c) if (nr, nc) in blue_cells])
            
            for r, c in blue_cells:
                if (r, c) not in connected:
                    path = [(r, c)]
                    while path[-1] not in connected:
                        nr, nc = min(get_neighbors(*path[-1]), key=lambda x: (x not in connected, random.random()))
                        path.append((nr, nc))
                    for pr, pc in path:
                        output_grid.set_cell(pr, pc, 1)
                        connected.add((pr, pc))

        # Fine-tune patterns
        for r, c in region:
            if output_grid.get_cell(r, c) == 1:
                green_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if output_grid.get_cell(nr, nc) == 3)
                if green_neighbors > 5 and random.random() < 0.3:
                    output_grid.set_cell(r, c, 3)
            elif output_grid.get_cell(r, c) == 3:
                blue_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if output_grid.get_cell(nr, nc) == 1)
                if blue_neighbors > 6 and random.random() < 0.7:
                    output_grid.set_cell(r, c, 1)

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
            # Ensure at least one green cell remains if possible
            if all(output_grid.get_cell(r, c) != 3 for r, c in region):
                if len(region) > 1:
                    r, c = random.choice([cell for cell in region if input_grid.get_cell(*cell) != 2])
                    output_grid.set_cell(r, c, 3)
        else:
            create_blue_pattern(region)

    return output_grid
