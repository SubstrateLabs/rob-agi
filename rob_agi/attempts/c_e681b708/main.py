from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Optional
from collections import deque

def solve_e681b708(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the main structure (largest connected component of blue cells).
    2. Finds endpoints of the structure and their colors.
    3. Creates a distance map from these endpoints.
    4. Propagates colors from endpoints to scattered blue dots based on distance.
    5. Merges adjacent transformed dots of the same color.
    6. Maintains the integrity of the main structure and original colored endpoints.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    main_structure = find_main_structure(input_grid)
    endpoints = find_endpoints(input_grid, main_structure)
    reachability_map = create_reachability_map(input_grid, main_structure, endpoints)
    
    output_grid = input_grid.deep_copy()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1 and (r, c) not in main_structure:
                if reachability_map[r][c] is not None:
                    output_grid.values[r][c] = reachability_map[r][c]
    
    merge_adjacent_dots(output_grid, main_structure)
    return output_grid

def find_main_structure(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    main_structure = set()
    
    def dfs(r: int, c: int, component: Set[Tuple[int, int]]):
        if (r, c) in visited or grid.values[r][c] != 1:
            return
        visited.add((r, c))
        component.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, component)
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1 and (r, c) not in visited:
                component = set()
                dfs(r, c, component)
                if len(component) > len(main_structure):
                    main_structure = component
    
    return main_structure

def find_endpoints(grid: ColoredGrid, structure: Set[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    rows, cols = grid.get_dimensions()
    endpoints = []
    for r, c in structure:
        neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) in structure)
        if neighbors == 1 or grid.values[r][c] != 1:
            endpoints.append((r, c, grid.values[r][c]))
    return endpoints

def create_distance_map(grid: ColoredGrid, structure: Set[Tuple[int, int]], endpoints: List[Tuple[int, int, int]]) -> List[List[List[Tuple[int, int]]]]:
    rows, cols = grid.get_dimensions()
    distance_map = [[[] for _ in range(cols)] for _ in range(rows)]
    
    for idx, (r, c, color) in enumerate(endpoints):
        queue = deque([(r, c, 0)])
        visited = set()
        while queue:
            cr, cc, dist = queue.popleft()
            if (cr, cc) in visited or (cr, cc) in structure:
                continue
            visited.add((cr, cc))
            distance_map[cr][cc].append((dist, idx))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in structure:
                    queue.append((nr, nc, dist + 1))
    
    return distance_map

def propagate_colors(grid: ColoredGrid, structure: Set[Tuple[int, int]], distance_map: List[List[List[Tuple[int, int]]]], endpoints: List[Tuple[int, int, int]]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_grid = grid.deep_copy()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in structure and grid.values[r][c] == 1:
                if distance_map[r][c]:
                    min_dist = min(dist for dist, _ in distance_map[r][c])
                    closest_endpoints = [idx for dist, idx in distance_map[r][c] if dist == min_dist]
                    if len(closest_endpoints) == 1:
                        new_grid.values[r][c] = endpoints[closest_endpoints[0]][2]
                    else:
                        # Priority system: prefer non-blue colors, then lower color values
                        colors = [endpoints[idx][2] for idx in closest_endpoints]
                        non_blue_colors = [color for color in colors if color != 1]
                        if non_blue_colors:
                            new_grid.values[r][c] = min(non_blue_colors)
                        else:
                            new_grid.values[r][c] = min(colors)
    
    return new_grid

def merge_adjacent_dots(grid: ColoredGrid, structure: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    visited = set()
    
    def dfs_merge(r: int, c: int, color: int):
        if (r, c) in visited or (r, c) in structure or grid.values[r][c] != color:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs_merge(nr, nc, color)
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and (r, c) not in structure and grid.values[r][c] != 1:
                dfs_merge(r, c, grid.values[r][c])

def solve_e681b708(input_grid: ColoredGrid) -> ColoredGrid:
    main_structure = find_main_structure(input_grid)
    endpoints = find_endpoints(input_grid, main_structure)
    distance_map = create_distance_map(input_grid, main_structure, endpoints)
    transformed_grid = propagate_colors(input_grid, main_structure, distance_map, endpoints)
    merge_adjacent_dots(transformed_grid, main_structure)
    return transformed_grid
