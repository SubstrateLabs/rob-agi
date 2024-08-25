from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Identifies connected regions of the same color.
    2. Applies color cycling: Blue (1) -> Sky Blue (8) -> Orange (7) -> Blue (1)
    3. Ensures no two adjacent regions end up with the same color (except blue).
    4. Preserves the overall structure and patterns of the input grid.
    """
    # Step 1: Identify connected regions
    regions = find_connected_regions(input_grid)
    
    # Step 2: Create a graph representation of adjacent regions
    region_graph = create_region_graph(regions)
    
    # Step 3: Apply color cycling logic
    new_colors = apply_color_cycling(regions, region_graph)
    
    # Step 4: Create and return the new grid
    return create_new_grid(input_grid, regions, new_colors)

def find_connected_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = []
                color = grid.values[r][c]
                queue = deque([(r, c)])
                while queue:
                    curr_r, curr_c = queue.popleft()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols:
                                queue.append((new_r, new_c))
                regions.append(region)
    return regions

def create_region_graph(regions: List[List[Tuple[int, int]]]) -> Dict[int, List[int]]:
    graph = {i: [] for i in range(len(regions))}
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions[i+1:], i+1):
            if are_adjacent(region1, region2):
                graph[i].append(j)
                graph[j].append(i)
    return graph

def are_adjacent(region1: List[Tuple[int, int]], region2: List[Tuple[int, int]]) -> bool:
    set1 = set(region1)
    for r, c in region2:
        if any((r+dr, c+dc) in set1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
            return True
    return False

def apply_color_cycling(regions: List[List[Tuple[int, int]]], graph: Dict[int, List[int]]) -> List[int]:
    new_colors = []
    for i, region in enumerate(regions):
        color = region[0][1]  # Get color of the region
        if color == 8:  # Sky blue always changes to orange
            new_colors.append(7)
        elif color == 7:  # Orange always changes to blue
            new_colors.append(1)
        else:  # Blue can stay blue or change to sky blue
            adjacent_colors = [new_colors[j] for j in graph[i] if j < i]
            if 8 in adjacent_colors or 7 in adjacent_colors:
                new_colors.append(8)
            else:
                new_colors.append(1)
    return new_colors

def create_new_grid(input_grid: ColoredGrid, regions: List[List[Tuple[int, int]]], new_colors: List[int]) -> ColoredGrid:
    new_grid = input_grid.deep_copy()
    for region, color in zip(regions, new_colors):
        for r, c in region:
            new_grid.values[r][c] = color
    return new_grid
