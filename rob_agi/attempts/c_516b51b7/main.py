from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_516b51b7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a pattern to connected blue regions.
    
    The solution follows these steps:
    1. Find all connected blue (1) regions in the input grid.
    2. For each region, determine its complexity level based on size.
    3. Apply a layering system to color cells based on their distance from the edge.
    4. Handle the center cells separately based on the region's complexity.
    
    The complexity levels and corresponding color patterns are:
    - Small (area < 16): Blue edge, Red interior
    - Medium (16 <= area < 36): Blue edge, Red inner layer, Green center
    - Large (area >= 36): Blue edge, alternating Red and Green layers, with special center handling
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    def find_connected_regions(color: int) -> List[List[Tuple[int, int]]]:
        return input_grid.find_connected_regions(color)
    
    def get_complexity_level(region: List[Tuple[int, int]]) -> int:
        area = len(region)
        if area < 16:
            return 1
        elif area < 36:
            return 2
        else:
            return 3
    
    def get_distance_from_edge(region: List[Tuple[int, int]]) -> dict:
        distances = {}
        queue = deque()
        for r, c in region:
            if any(0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 0
                   for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]):
                distances[(r, c)] = 0
                queue.append((r, c, 0))
        
        while queue:
            r, c, d = queue.popleft()
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if (nr, nc) in region and (nr, nc) not in distances:
                    distances[(nr, nc)] = d + 1
                    queue.append((nr, nc, d + 1))
        
        return distances
    
    def color_region(region: List[Tuple[int, int]], complexity: int):
        distances = get_distance_from_edge(region)
        max_distance = max(distances.values())
        center_distance = max_distance if complexity < 3 else (max_distance // 2) + (max_distance % 2)
        
        for r, c in region:
            d = distances[(r, c)]
            if d == center_distance:
                output_grid.values[r][c] = 3 if complexity > 1 else 2
            elif complexity == 1:
                output_grid.values[r][c] = 2 if d > 0 else 1
            elif complexity == 2:
                output_grid.values[r][c] = 3 if d > 1 else (2 if d > 0 else 1)
            else:
                output_grid.values[r][c] = [1, 2, 3][(d - 1) % 3] if d > 0 else 1
    
    blue_regions = find_connected_regions(1)
    for region in blue_regions:
        complexity = get_complexity_level(region)
        color_region(region, complexity)
    
    return output_grid
