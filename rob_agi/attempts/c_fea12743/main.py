from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_fea12743(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the fea12743 challenge by identifying distinct colored regions,
    analyzing their adjacencies, and applying color transformations based on
    the main pattern. The solution maintains the original black cells and
    transforms the colored regions according to the pattern:
    - The main connected component is identified
    - Within it, the most connected region remains red
    - One adjacent region becomes green
    - Other regions in the main component become sky blue
    - Disconnected regions remain red
    """
    # Step 1: Identify distinct regions
    regions = find_regions(input_grid)
    
    # Step 2 & 3: Analyze region adjacencies and identify the main pattern
    adjacency_graph = build_adjacency_graph(regions)
    main_component = find_largest_component(adjacency_graph)
    
    # Step 4: Select the starting red region
    start_region = max(main_component, key=lambda r: len(adjacency_graph[r]))
    
    # Step 5: Apply color transformations
    new_grid = ColoredGrid(values=[[0 for _ in range(input_grid.get_dimensions()[1])] 
                                   for _ in range(input_grid.get_dimensions()[0])])
    
    green_region = next(iter(adjacency_graph[start_region]))
    
    for region in regions:
        if region == start_region:
            color = 2  # red
        elif region == green_region:
            color = 3  # green
        elif region in main_component:
            color = 8  # sky blue
        else:
            color = 2  # disconnected regions remain red
        
        for x, y in region['cells']:
            new_grid.values[x][y] = color
    
    # Step 7: Preserve black cells
    for x in range(input_grid.get_dimensions()[0]):
        for y in range(input_grid.get_dimensions()[1]):
            if input_grid.values[x][y] == 0:
                new_grid.values[x][y] = 0
    
    return new_grid

def build_adjacency_graph(regions):
    graph = {r: set() for r in regions}
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            if are_adjacent(r1, r2):
                graph[r1].add(r2)
                graph[r2].add(r1)
    return graph

def are_adjacent(region1, region2):
    for x1, y1 in region1['cells']:
        for x2, y2 in region2['cells']:
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                return True
    return False

def find_largest_component(graph):
    visited = set()
    largest_component = []
    
    def dfs(node):
        component = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                stack.extend(graph[current] - visited)
        return component
    
    for node in graph:
        if node not in visited:
            component = dfs(node)
            if len(component) > len(largest_component):
                largest_component = component
    
    return largest_component

def find_regions(grid: ColoredGrid) -> List[Dict]:
    regions = []
    visited = set()
    for x in range(grid.get_dimensions()[0]):
        for y in range(grid.get_dimensions()[1]):
            if (x, y) not in visited and grid.values[x][y] != 0:
                region = flood_fill(grid, x, y, grid.values[x][y])
                regions.append({
                    'color': grid.values[x][y],
                    'cells': region,
                    'centroid': calculate_centroid(region)
                })
                visited.update(region)
    return regions

def flood_fill(grid: ColoredGrid, x: int, y: int, color: int) -> List[Tuple[int, int]]:
    cells = []
    stack = [(x, y)]
    visited = set()
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited or grid.values[cx][cy] != color:
            continue
        visited.add((cx, cy))
        cells.append((cx, cy))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid.get_dimensions()[0] and 0 <= ny < grid.get_dimensions()[1]:
                stack.append((nx, ny))
    return cells

def calculate_centroid(cells: List[Tuple[int, int]]) -> Tuple[float, float]:
    return sum(x for x, _ in cells) / len(cells), sum(y for _, y in cells) / len(cells)

def order_regions(regions: List[Dict], grid_dimensions: Tuple[int, int]) -> List[Dict]:
    center_x, center_y = grid_dimensions[0] / 2, grid_dimensions[1] / 2
    for region in regions:
        cx, cy = region['centroid']
        if cx < center_x and cy < center_y:
            region['position'] = 'top-left'
        elif cx < center_x and cy >= center_y:
            region['position'] = 'bottom-left'
        elif cx >= center_x and cy < center_y:
            region['position'] = 'top-right'
        else:
            region['position'] = 'bottom-right'
    return sorted(regions, key=lambda r: ['top-left', 'top-right', 'bottom-right', 'bottom-left'].index(r['position']))
