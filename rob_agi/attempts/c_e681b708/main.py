from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_e681b708(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the main structure (largest connected component of blue cells).
    2. Finds endpoints of the structure and their colors.
    3. Creates a reachability map from these endpoints.
    4. Transforms blue dots not part of the structure based on reachability.
    5. Merges adjacent transformed dots of the same color.

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

def create_reachability_map(grid: ColoredGrid, structure: Set[Tuple[int, int]], endpoints: List[Tuple[int, int, int]]) -> List[List[Optional[int]]]:
    rows, cols = grid.get_dimensions()
    reachability_map = [[None for _ in range(cols)] for _ in range(rows)]
    
    for r, c, color in endpoints:
        queue = deque([(r, c)])
        visited = set()
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in visited or (cr, cc) in structure:
                continue
            visited.add((cr, cc))
            reachability_map[cr][cc] = color
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in structure:
                    queue.append((nr, nc))
    
    return reachability_map

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
