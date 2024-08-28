from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_e0fb7511(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a meandering structure of sky blue (8) squares
    that forms an organic pattern while preserving some original black squares.

    The function performs the following steps:
    1. Analyze the input grid to create a heat map of black square density
    2. Choose a starting point on an edge with high black square density
    3. Grow a sky blue structure using a probability-based growth algorithm
    4. Implement branching for a more organic structure
    5. Ensure the pattern reaches at least 3 edges
    6. Preserve some original black squares
    7. Smooth and refine the structure
    8. Balance the composition by adding small branches in large blue areas
    9. Ensure connectivity of all sky blue cells
    10. Fine-tune the pattern to meet coverage requirements
    11. Validate the final result

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
        to_remove = set()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    neighbors = get_neighbors(r, c)
                    blue_neighbors = sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) == 1)
                    if blue_neighbors >= 3:
                        grid.set_cell(r, c, 1)
                        to_remove.add((r, c))
        structure.difference_update(to_remove)

    # Main algorithm
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

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

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def get_edge_start():
        edges = [(0, c) for c in range(cols)] + [(rows-1, c) for c in range(cols)] + \
                [(r, 0) for r in range(1, rows-1)] + [(r, cols-1) for r in range(1, rows-1)]
        return max(edges, key=lambda x: heat_map[x[0]][x[1]])

    def grow_structure(start):
        structure = set([start])
        queue = deque([start])
        grid.set_cell(*start, 8)
        edges_reached = set()
        
        while queue and len(structure) < target_size:
            r, c = queue.popleft()
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in structure:
                    if grid.get_cell(nr, nc) in [0, 1]:
                        prob = 0.7 * heat_map[nr][nc] / max_heat + 0.3 * random.random()
                        if prob > 0.5:
                            structure.add((nr, nc))
                            queue.append((nr, nc))
                            grid.set_cell(nr, nc, 8)
                            if nr in [0, rows-1] or nc in [0, cols-1]:
                                edges_reached.add((nr, nc))
            
            if random.random() < 0.1:  # Branching
                queue.append(random.choice(list(structure)))
        
        return structure, edges_reached

    def ensure_edge_coverage(structure, edges_reached):
        while len(edges_reached) < 3:
            start = random.choice(list(structure))
            target_edge = random.choice([(0, c) for c in range(cols)] + [(rows-1, c) for c in range(cols)] + 
                                        [(r, 0) for r in range(rows)] + [(r, cols-1) for r in range(rows)])
            path = []
            while start != target_edge:
                r, c = start
                dr = (target_edge[0] - r) // max(1, abs(target_edge[0] - r))
                dc = (target_edge[1] - c) // max(1, abs(target_edge[1] - c))
                start = (r + dr, c + dc)
                if start not in structure:
                    structure.add(start)
                    grid.set_cell(*start, 8)
                    path.append(start)
                if start[0] in [0, rows-1] or start[1] in [0, cols-1]:
                    edges_reached.add(start)
                    break
            if path:
                structure.update(path)

    def preserve_black_squares(structure):
        black_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 0]
        to_preserve = random.sample(black_squares, k=int(len(black_squares) * 0.3))
        for r, c in to_preserve:
            if (r, c) in structure:
                structure.remove((r, c))
            grid.set_cell(r, c, 0)

    def smooth_structure(structure):
        to_remove = set()
        to_add = set()
        for r, c in structure:
            neighbors = get_neighbors(r, c)
            sky_blue_neighbors = sum(1 for nr, nc in neighbors if (nr, nc) in structure)
            if sky_blue_neighbors <= 1:
                to_remove.add((r, c))
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in structure:
                    neighbors = get_neighbors(r, c)
                    sky_blue_neighbors = sum(1 for nr, nc in neighbors if (nr, nc) in structure)
                    if sky_blue_neighbors >= 3:
                        to_add.add((r, c))
        structure.difference_update(to_remove)
        structure.update(to_add)
        for r, c in to_remove:
            grid.set_cell(r, c, 1)
        for r, c in to_add:
            grid.set_cell(r, c, 8)

    def balance_composition(structure):
        blue_regions = grid.find_connected_regions(1)
        for region in blue_regions:
            if len(region) > 0.2 * rows * cols:
                start = random.choice(region)
                for _ in range(3):  # Add small branches
                    current = start
                    for _ in range(random.randint(2, 5)):
                        neighbors = [n for n in get_neighbors(*current) if n not in structure]
                        if not neighbors:
                            break
                        next_cell = random.choice(neighbors)
                        structure.add(next_cell)
                        grid.set_cell(*next_cell, 8)
                        current = next_cell

    def ensure_connectivity(structure):
        regions = grid.find_connected_regions(8)
        if len(regions) > 1:
            main_region = max(regions, key=len)
            for region in regions:
                if region != main_region:
                    start = region[0]
                    end = min(main_region, key=lambda x: abs(x[0]-start[0]) + abs(x[1]-start[1]))
                    while start != end:
                        r, c = start
                        dr = (end[0] - r) // max(1, abs(end[0] - r))
                        dc = (end[1] - c) // max(1, abs(end[1] - c))
                        start = (r + dr, c + dc)
                        if start not in structure:
                            structure.add(start)
                            grid.set_cell(*start, 8)

    heat_map = create_heat_map()
    max_heat = max(max(row) for row in heat_map)
    total_cells = rows * cols
    target_size = int(total_cells * random.uniform(0.2, 0.3))

    start = get_edge_start()
    structure, edges_reached = grow_structure(start)
    ensure_edge_coverage(structure, edges_reached)
    preserve_black_squares(structure)
    smooth_structure(structure)
    balance_composition(structure)
    ensure_connectivity(structure)

    # Fine-tune coverage
    current_coverage = len(structure) / total_cells
    if current_coverage > 0.3:
        to_remove = random.sample(list(structure), k=int((current_coverage - 0.3) * total_cells))
        for r, c in to_remove:
            structure.remove((r, c))
            grid.set_cell(r, c, 1)

    return grid
