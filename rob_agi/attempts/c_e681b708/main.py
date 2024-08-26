from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_e681b708(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the main structure (largest connected component of blue cells).
    2. Finds endpoints of the structure and their colors.
    3. Divides the grid into regions based on the main structure.
    4. Assigns colors to regions in a specific order: red (2), green (3), sky blue (8).
    5. Transforms scattered blue dots to the color of their region.
    6. Merges adjacent transformed dots of the same color.
    7. Handles special cases where endpoint colors influence nearby areas.
    8. Maintains the integrity of the main structure and original colored endpoints.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    main_structure = find_main_structure(input_grid)
    endpoints = find_endpoints(input_grid, main_structure)
    regions = divide_into_regions(input_grid, main_structure)
    region_colors = assign_colors_to_regions(regions, endpoints)
    
    output_grid = input_grid.deep_copy()
    transform_scattered_dots(output_grid, main_structure, region_colors)
    handle_special_cases(output_grid, endpoints, main_structure)
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

def divide_into_regions(grid: ColoredGrid, structure: Set[Tuple[int, int]]) -> Dict[int, Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    regions = {}
    region_id = 0
    visited = set(structure)
    
    def flood_fill(r: int, c: int, region: Set[Tuple[int, int]]):
        if (r, c) in visited:
            return
        visited.add((r, c))
        region.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in structure:
                flood_fill(nr, nc, region)
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                new_region = set()
                flood_fill(r, c, new_region)
                regions[region_id] = new_region
                region_id += 1
    
    return regions

def assign_colors_to_regions(regions: Dict[int, Set[Tuple[int, int]]], endpoints: List[Tuple[int, int, int]]) -> Dict[int, int]:
    colors = [2, 3, 8]  # red, green, sky blue
    region_colors = {}
    for i, region_id in enumerate(regions.keys()):
        region_colors[region_id] = colors[i % len(colors)]
    return region_colors

def transform_scattered_dots(grid: ColoredGrid, structure: Set[Tuple[int, int]], region_colors: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1 and (r, c) not in structure:
                region_id = next(rid for rid, region in region_colors.items() if (r, c) in region)
                grid.values[r][c] = region_colors[region_id]

def handle_special_cases(grid: ColoredGrid, endpoints: List[Tuple[int, int, int]], structure: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for er, ec, color in endpoints:
        if color != 1:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = er + dr, ec + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in structure:
                    grid.values[nr][nc] = color

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
