from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_e681b708(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the main structure (connected blue cells and colored endpoints).
    2. Divides the grid into regions based on the main structure.
    3. Analyzes each region's position relative to the structure and endpoints.
    4. Assigns colors to regions based on their position and nearest endpoints.
    5. Transforms scattered blue dots to the color of their region.
    6. Handles special cases:
       - Keeps blue dots below the bottommost horizontal line blue.
       - Colors cells adjacent to endpoints appropriately.
    7. Preserves the main structure and original colored endpoints.

    Color assignment rules:
    - Above the structure: Red (2)
    - Below or to the sides of the structure: Green (3)
    - Sky Blue (8) is used when it's the nearest endpoint color, overriding other rules.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    main_structure, endpoints = find_main_structure_and_endpoints(input_grid)
    regions = divide_into_regions(input_grid, main_structure)
    region_colors = assign_colors_to_regions(input_grid, regions, endpoints, main_structure)
    
    output_grid = input_grid.deep_copy()
    transform_scattered_dots(output_grid, main_structure, region_colors)
    handle_special_cases(output_grid, endpoints, main_structure)
    preserve_main_structure(output_grid, main_structure, input_grid)
    
    return output_grid

def find_main_structure_and_endpoints(grid: ColoredGrid) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int, int]]]:
    rows, cols = grid.get_dimensions()
    structure = set()
    endpoints = []
    
    def dfs(r: int, c: int):
        if (r, c) in structure or grid.values[r][c] not in [1, 2, 3, 6, 8]:
            return
        structure.add((r, c))
        if grid.values[r][c] != 1:
            endpoints.append((r, c, grid.values[r][c]))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc)
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] in [1, 2, 3, 6, 8] and (r, c) not in structure:
                dfs(r, c)
    
    return structure, endpoints

def divide_into_regions(grid: ColoredGrid, structure: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    regions = []
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
                regions.append(new_region)
    
    return regions

def assign_colors_to_regions(grid: ColoredGrid, regions: List[Set[Tuple[int, int]]], endpoints: List[Tuple[int, int, int]], structure: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    rows, cols = grid.get_dimensions()
    region_colors = {}
    
    def distance_to_endpoint(r: int, c: int, endpoint: Tuple[int, int, int]) -> float:
        return ((r - endpoint[0])**2 + (c - endpoint[1])**2)**0.5
    
    structure_top = min(r for r, _ in structure)
    structure_bottom = max(r for r, _ in structure)
    structure_left = min(c for _, c in structure)
    structure_right = max(c for _, c in structure)
    
    for region in regions:
        center_r = sum(r for r, _ in region) / len(region)
        center_c = sum(c for _, c in region) / len(region)
        
        nearest_endpoint = min(endpoints, key=lambda e: distance_to_endpoint(center_r, center_c, e))
        
        if nearest_endpoint[2] == 8:
            color = 8  # Sky Blue
        elif center_r < structure_top:
            color = 2  # Red
        else:
            color = 3  # Green
        
        for r, c in region:
            region_colors[(r, c)] = color
    
    return region_colors

def transform_scattered_dots(grid: ColoredGrid, structure: Set[Tuple[int, int]], region_colors: Dict[Tuple[int, int], int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in structure and grid.values[r][c] == 1:
                grid.values[r][c] = region_colors.get((r, c), 1)

def handle_special_cases(grid: ColoredGrid, endpoints: List[Tuple[int, int, int]], structure: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    bottom_line = max(r for r, _ in structure)
    
    for r in range(bottom_line + 1, rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                continue  # Keep blue dots below the bottommost line blue
    
    for er, ec, color in endpoints:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = er + dr, ec + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in structure:
                grid.values[nr][nc] = color

def preserve_main_structure(output_grid: ColoredGrid, structure: Set[Tuple[int, int]], input_grid: ColoredGrid):
    for r, c in structure:
        output_grid.values[r][c] = input_grid.values[r][c]
