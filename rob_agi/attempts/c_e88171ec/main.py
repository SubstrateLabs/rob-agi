from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_e88171ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e88171ec challenge by finding a suitable region of black cells
    and filling it with sky blue (8).

    The solution follows these steps:
    1. Find all contiguous regions of black (0) cells.
    2. Filter regions to those at least 2x2 in size.
    3. Rank regions based on size and centrality.
    4. Select the best region and determine the fill area (up to 4x4).
    5. Fill the chosen area with sky blue (8).

    If no suitable region is found, return the original grid unchanged.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find all black regions
    black_regions = find_black_regions(input_grid)
    
    # Filter and rank regions
    suitable_regions = [region for region in black_regions if len(region) >= 4]
    if not suitable_regions:
        return output_grid
    
    ranked_regions = rank_regions(suitable_regions, rows, cols)
    best_region = ranked_regions[0]
    
    # Determine fill area
    fill_area = determine_fill_area(best_region, rows, cols)
    
    # Fill the area with sky blue (8)
    for r, c in fill_area:
        output_grid.set_cell(r, c, 8)
    
    return output_grid

def find_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                region = []
                queue = deque([(r, c)])
                while queue:
                    curr_r, curr_c = queue.popleft()
                    if (curr_r, curr_c) not in visited:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols and grid.get_cell(new_r, new_c) == 0:
                                queue.append((new_r, new_c))
                regions.append(region)
    return regions

def rank_regions(regions: List[List[Tuple[int, int]]], rows: int, cols: int) -> List[List[Tuple[int, int]]]:
    def region_score(region):
        size = len(region)
        center_r = sum(r for r, _ in region) / size
        center_c = sum(c for _, c in region) / size
        distance_from_center = ((center_r - rows/2)**2 + (center_c - cols/2)**2)**0.5
        return size - distance_from_center  # Simple scoring function

    return sorted(regions, key=region_score, reverse=True)

def determine_fill_area(region: List[Tuple[int, int]], rows: int, cols: int) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    if height > 4 or width > 4:
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2
        min_r = max(0, center_r - 1)
        max_r = min(rows - 1, center_r + 2)
        min_c = max(0, center_c - 1)
        max_c = min(cols - 1, center_c + 2)
    
    return [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
