from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_e0fb7511(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a meandering structure of sky blue (8) squares
    that starts from a random edge, grows towards areas with more black squares,
    and forms an organic pattern while preserving some original black squares.

    The function performs the following steps:
    1. Analyze the input grid to create a heat map of black square density
    2. Choose a random starting point on an edge, biased towards higher density areas
    3. Grow a sky blue structure using a probability-based growth algorithm
    4. Implement branching for a more organic structure
    5. Control growth direction based on the black square heat map
    6. Ensure connectivity of all sky blue cells
    7. Smooth and refine the structure
    8. Preserve some original black squares
    9. Balance the final composition
    10. Validate and adjust the result

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with the sky blue structure pattern
    """
    from collections import deque
    import random
    import math

    def create_heat_map():
        heat_map = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if 0 <= r+dr < rows and 0 <= c+dc < cols:
                                heat_map[r+dr][c+dc] += 1 / (1 + math.sqrt(dr**2 + dc**2))
        return heat_map

    def get_random_edge_start():
        edges = (
            [(0, c) for c in range(cols)] +  # Top edge
            [(rows-1, c) for c in range(cols)] +  # Bottom edge
            [(r, 0) for r in range(1, rows-1)] +  # Left edge
            [(r, cols-1) for r in range(1, rows-1)]  # Right edge
        )
        weights = [heat_map[r][c] for r, c in edges]
        return random.choices(edges, weights=weights)[0]

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def grow_structure(start):
        queue = deque([start])
        structure = set([start])
        grid.set_cell(*start, 8)
        direction = random.uniform(0, 2*math.pi)
        steps = 0
        
        while queue and len(structure) < target_size:
            r, c = queue.popleft()
            neighbors = get_neighbors(r, c)
            random.shuffle(neighbors)
            for nr, nc in neighbors:
                if (nr, nc) not in structure:
                    cell_value = grid.get_cell(nr, nc)
                    prob = growth_probability(nr, nc, direction)
                    if cell_value == 0 or (cell_value == 1 and random.random() < prob):
                        structure.add((nr, nc))
                        queue.append((nr, nc))
                        grid.set_cell(nr, nc, 8)
            
            steps += 1
            if steps % 10 == 0:
                direction = update_direction(direction, r, c)
                if random.random() < branching_probability:
                    branch_start = random.choice(list(structure))
                    queue.append(branch_start)
        
        return structure

    def growth_probability(r, c, direction):
        base_prob = 0.2 + 0.6 * heat_map[r][c] / max_heat
        dir_bias = 0.2 * math.cos(math.atan2(c - cols/2, r - rows/2) - direction)
        return min(0.9, base_prob + dir_bias)

    def update_direction(direction, r, c):
        target_direction = math.atan2(c - cols/2, r - rows/2)
        return (0.9 * direction + 0.1 * target_direction) % (2*math.pi)

    def smooth_structure(structure):
        for r, c in list(structure):
            neighbors = get_neighbors(r, c)
            sky_blue_neighbors = sum(1 for nr, nc in neighbors if (nr, nc) in structure)
            if sky_blue_neighbors <= 1:
                structure.remove((r, c))
                grid.set_cell(r, c, grid.get_cell(r, c))
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in structure:
                    neighbors = get_neighbors(r, c)
                    sky_blue_neighbors = sum(1 for nr, nc in neighbors if (nr, nc) in structure)
                    if sky_blue_neighbors >= 3:
                        structure.add((r, c))
                        grid.set_cell(r, c, 8)

    def ensure_connectivity(structure):
        connected = set()
        stack = [next(iter(structure))]
        while stack:
            cell = stack.pop()
            if cell not in connected:
                connected.add(cell)
                stack.extend(n for n in get_neighbors(*cell) if n in structure and n not in connected)
        
        disconnected = structure - connected
        for cell in disconnected:
            path = shortest_path(cell, connected)
            for r, c in path:
                grid.set_cell(r, c, 8)
                structure.add((r, c))
                connected.add((r, c))

    def shortest_path(start, targets):
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) in targets:
                return path
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def preserve_black_squares(structure):
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0 and (r, c) not in structure:
                    neighbors = get_neighbors(r, c)
                    if not any((nr, nc) in structure for nr, nc in neighbors):
                        if random.random() < 0.8:
                            grid.set_cell(r, c, 0)
                    elif random.random() < 0.3:
                        grid.set_cell(r, c, 0)

    def balance_composition(structure):
        edges_reached = sum(1 for r, c in structure if r in (0, rows-1) or c in (0, cols-1))
        while edges_reached < 3:
            r, c = random.choice(list(structure))
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in structure:
                    structure.add((nr, nc))
                    grid.set_cell(nr, nc, 8)
                    if nr in (0, rows-1) or nc in (0, cols-1):
                        edges_reached += 1
                    break

    def fine_tune_pattern(structure):
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    neighbors = get_neighbors(r, c)
                    blue_neighbors = sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) == 1)
                    if blue_neighbors >= 3:
                        grid.set_cell(r, c, 1)
                        structure.remove((r, c))

    # Main algorithm
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    heat_map = create_heat_map()
    max_heat = max(max(row) for row in heat_map)
    
    total_cells = rows * cols
    target_size = int(total_cells * random.uniform(0.3, 0.4))  # Reduced target size
    branching_probability = 0.15  # Increased branching probability

    start = get_random_edge_start()
    structure = grow_structure(start)
    smooth_structure(structure)
    ensure_connectivity(structure)
    preserve_black_squares(structure)
    balance_composition(structure)
    fine_tune_pattern(structure)

    return grid
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Helper functions
    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_black(r: int, c: int) -> bool:
        return grid.get_cell(r, c) == 0

    def set_sky_blue(r: int, c: int):
        grid.set_cell(r, c, 8)

    # Analyze black square distribution
    black_squares = [(r, c) for r in range(rows) for c in range(cols) if is_black(r, c)]
    total_black = len(black_squares)

    # Choose a starting point
    start = max(black_squares, key=lambda x: sum(1 for nr, nc in get_neighbors(*x) if is_black(nr, nc)))
    path = [start]
    set_sky_blue(*start)

    # Grow the path
    target_length = int(total_black * random.uniform(0.5, 0.8))
    while len(path) < target_length:
        r, c = path[-1]
        neighbors = get_neighbors(r, c)
        valid_moves = [
            (nr, nc) for nr, nc in neighbors
            if grid.get_cell(nr, nc) in [0, 1] and (nr, nc) not in path
        ]
        if not valid_moves:
            # Implement jumping mechanism
            unvisited_black = [sq for sq in black_squares if sq not in path]
            if not unvisited_black:
                break
            jump_to = min(unvisited_black, key=lambda x: abs(x[0]-r) + abs(x[1]-c))
            while (r, c) != jump_to:
                r += (jump_to[0] - r) // max(1, abs(jump_to[0] - r))
                c += (jump_to[1] - c) // max(1, abs(jump_to[1] - c))
                set_sky_blue(r, c)
                path.append((r, c))
        else:
            next_move = max(valid_moves, key=lambda x: 2 if is_black(*x) else 1)
            set_sky_blue(*next_move)
            path.append(next_move)

    # Preserve original black squares
    black_to_keep = random.sample(black_squares, k=int(total_black * random.uniform(0.2, 0.4)))
    for r, c in black_to_keep:
        if (r, c) not in path:
            grid.set_cell(r, c, 0)

    # Fine-tune the pattern
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                neighbors = get_neighbors(r, c)
                sky_blue_neighbors = sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) == 8)
                if sky_blue_neighbors <= 1:
                    grid.set_cell(r, c, 1)  # Convert isolated sky blue to blue

    # Final connectivity check
    sky_blue_regions = grid.find_connected_regions(8)
    if len(sky_blue_regions) > 1:
        main_region = max(sky_blue_regions, key=len)
        for region in sky_blue_regions:
            if region != main_region:
                start = region[0]
                end = min(main_region, key=lambda x: abs(x[0]-start[0]) + abs(x[1]-start[1]))
                while start != end:
                    r, c = start
                    r += (end[0] - r) // max(1, abs(end[0] - r))
                    c += (end[1] - c) // max(1, abs(end[1] - c))
                    set_sky_blue(r, c)
                    start = (r, c)

    return grid
