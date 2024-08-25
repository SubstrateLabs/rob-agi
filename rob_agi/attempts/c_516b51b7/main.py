from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_516b51b7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a concentric pattern to connected blue regions.
    
    The solution follows these steps:
    1. Find all connected blue (1) regions in the input grid.
    2. For each region, calculate the distance of each cell from the edge.
    3. Apply a concentric coloring pattern based on the region size:
       - Small regions (3x3 or smaller): Keep edge blue (1), inner cells red (2)
       - Medium regions (4x4 or 4x3): Keep edge blue (1), inner cells red (2)
       - Large regions (5x5 or larger):
         * Edge (distance 0): Blue (1)
         * First inner layer (distance 1): Red (2)
         * Second inner layer (distance 2): Green (3)
         * Center:
           - For even dimensions: 2x2 red (2) center
           - For odd dimensions: Green (3) center
    
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
        
        if height <= 3 and width <= 3:  # Small region
            for r, c in region:
                output_grid.values[r][c] = 1 if distances[(r, c)] == 0 else 2
        elif (height == 4 and width <= 4) or (width == 4 and height <= 4):  # Medium region
            for r, c in region:
                output_grid.values[r][c] = 1 if distances[(r, c)] == 0 else 2
        else:  # Large region
            for r, c in region:
                d = distances[(r, c)]
                if d == 0:
                    output_grid.values[r][c] = 1  # Blue edge
                elif d == 1:
                    output_grid.values[r][c] = 2  # Red first inner layer
                elif d == 2:
                    output_grid.values[r][c] = 3  # Green second inner layer
                else:
                    center_r, center_c = min_r + height // 2, min_c + width // 2
                    if height % 2 == 0 and width % 2 == 0:
                        output_grid.values[r][c] = 2 if (abs(r - center_r) < 2 and abs(c - center_c) < 2) else 3
                    else:
                        output_grid.values[r][c] = 3
    
    blue_regions = find_connected_regions(1)
    for region in blue_regions:
        color_region(region)
    
    return output_grid
