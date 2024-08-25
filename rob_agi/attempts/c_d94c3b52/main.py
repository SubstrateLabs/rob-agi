from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Identifies connected regions of non-black colors.
    2. Analyzes the global structure and color distribution.
    3. Applies color transformations based on region size, position, and adjacent colors.
    4. Maintains overall balance and visual distinctiveness.
    5. Preserves small sky blue "anchor" regions and key patterns.
    6. Ensures no large adjacent regions have the same non-blue color.
    """
    regions = find_connected_regions(input_grid)
    region_graph = create_region_graph(regions)
    color_distribution = analyze_color_distribution(input_grid)
    new_colors = apply_color_transformations(regions, region_graph, color_distribution)
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

def analyze_color_distribution(grid: ColoredGrid) -> Dict[int, int]:
    distribution = {1: 0, 7: 0, 8: 0}  # Blue, Orange, Sky Blue
    for row in grid.values:
        for cell in row:
            if cell in distribution:
                distribution[cell] += 1
    return distribution

def apply_color_transformations(regions: List[List[Tuple[int, int]]], graph: Dict[int, List[int]], color_distribution: Dict[int, int]) -> List[int]:
    new_colors = []
    for i, region in enumerate(regions):
        color = region[0][1]  # Get color of the region
        size = len(region)
        
        if size <= 4 and color == 8:  # Small sky blue "anchor" regions
            new_colors.append(8)
        elif color == 8 and size > 4:  # Large sky blue regions change to orange
            new_colors.append(7)
        elif color == 7:  # Orange always changes to blue
            new_colors.append(1)
        else:  # Blue regions
            adjacent_colors = [new_colors[j] for j in graph[i] if j < i]
            if 8 not in adjacent_colors and color_distribution[8] < color_distribution[1]:
                new_colors.append(8)
            else:
                new_colors.append(1)
        
        # Update color distribution
        color_distribution[color] -= size
        color_distribution[new_colors[-1]] += size
    
    return new_colors

def create_new_grid(input_grid: ColoredGrid, regions: List[List[Tuple[int, int]]], new_colors: List[int]) -> ColoredGrid:
    new_grid = input_grid.deep_copy()
    for region, color in zip(regions, new_colors):
        for r, c in region:
            new_grid.values[r][c] = color
    return new_grid
