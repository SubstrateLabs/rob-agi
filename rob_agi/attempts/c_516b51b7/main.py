from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_516b51b7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a concentric pattern to connected blue regions.
    
    The solution follows these steps:
    1. Find all connected blue (1) regions in the input grid.
    2. For each region, calculate the distance of each cell from the edge.
    3. Apply a concentric coloring pattern based on the distance from the edge:
       - Edge (distance 0): Blue (1)
       - First inner layer (distance 1): Red (2)
       - Second inner layer and beyond: Alternating Green (3) and Red (2)
    4. For larger regions, apply a special center treatment:
       - For regions with even dimensions, create a 2x2 green center
       - For regions with odd dimensions, keep the center as determined by the concentric pattern
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    def find_connected_regions(color: int) -> List[List[Tuple[int, int]]]:
        return input_grid.find_connected_regions(color)
    
    def get_distance_from_edge(region: List[Tuple[int, int]]) -> dict:
        distances = {}
        queue = deque()
        for r, c in region:
            if any(0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 1
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
    
    def get_region_dimensions(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1
    
    def color_region(region: List[Tuple[int, int]]):
        distances = get_distance_from_edge(region)
        min_r, min_c, height, width = get_region_dimensions(region)
        max_distance = max(distances.values())
        
        for r, c in region:
            d = distances[(r, c)]
            if d == 0:
                output_grid.values[r][c] = 1  # Blue edge
            elif d == 1:
                output_grid.values[r][c] = 2  # Red first inner layer
            else:
                output_grid.values[r][c] = 3 if d % 2 == 0 else 2  # Alternating Green and Red
        
        # Special center treatment for larger regions
        if max_distance >= 2 and (height >= 4 or width >= 4):
            center_r = min_r + height // 2
            center_c = min_c + width // 2
            if height % 2 == 0 and width % 2 == 0:
                for dr in range(2):
                    for dc in range(2):
                        output_grid.values[center_r - 1 + dr][center_c - 1 + dc] = 3  # Green 2x2 center
    
    blue_regions = find_connected_regions(1)
    for region in blue_regions:
        color_region(region)
    
    return output_grid
