from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_516b51b7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a concentric pattern to connected blue regions.
    
    The solution follows these steps:
    1. Find all connected blue (1) regions in the input grid.
    2. For each region:
       a. Calculate its dimensions and center point(s).
       b. For each cell, calculate its distance from the center.
    3. Apply a concentric coloring pattern based on the region size:
       - Small regions (3x3 or smaller): Keep edge blue (1), inner cells red (2)
       - Medium and large regions (4x4 or larger):
         * Center: Red (2) for 2x2 center if even dimensions, Green (3) if odd
         * Next layer (if exists): Green (3)
         * Following layer: Red (2)
         * Edge: Blue (1)
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    def find_connected_regions(color: int) -> List[List[Tuple[int, int]]]:
        return input_grid.find_connected_regions(color)
    
    def get_region_info(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int, Tuple[float, float]]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        center_r = min_r + (height - 1) / 2
        center_c = min_c + (width - 1) / 2
        return min_r, min_c, height, width, (center_r, center_c)
    
    def color_region(region: List[Tuple[int, int]]):
        min_r, min_c, height, width, (center_r, center_c) = get_region_info(region)
        max_distance = max(height, width) // 2
        
        for r, c in region:
            distance = max(abs(r - center_r), abs(c - center_c))
            if height <= 3 and width <= 3:  # Small region
                output_grid.values[r][c] = 1 if distance == max_distance else 2
            else:  # Medium and large regions
                if distance == max_distance:
                    output_grid.values[r][c] = 1  # Blue edge
                elif distance == max_distance - 1:
                    output_grid.values[r][c] = 2  # Red layer
                elif distance == max_distance - 2 and max_distance > 2:
                    output_grid.values[r][c] = 3  # Green layer
                else:
                    # Center coloring
                    if height % 2 == 0 and width % 2 == 0:
                        output_grid.values[r][c] = 2 if (abs(r - center_r) < 1 and abs(c - center_c) < 1) else 3
                    else:
                        output_grid.values[r][c] = 3
    
    blue_regions = find_connected_regions(1)
    for region in blue_regions:
        color_region(region)
    
    return output_grid
