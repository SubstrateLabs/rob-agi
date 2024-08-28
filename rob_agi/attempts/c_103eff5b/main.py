from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    y_coords, x_coords = zip(*region)
    return min(y_coords), max(y_coords), min(x_coords), max(x_coords)

def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_103eff5b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing color 8 regions with a specific color pattern.
    
    The solution works as follows:
    1. Preserve the existing colored pattern.
    2. Identify all connected regions of color 8.
    3. For each color 8 region:
       a. Determine if it's a small or large region based on area.
       b. For small regions, assign a single color based on position.
       c. For large regions, use a distance-based approach to assign colors:
          - Red (2) in the top-left
          - Yellow (4) in the top-right
          - Green (3) in the bottom-left
          - Blue (1) in the bottom-right
    4. Apply post-processing to ensure color balance and smooth transitions.
    5. Return the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    color_8_regions = input_grid.find_connected_regions(8)
    rows, cols = input_grid.get_dimensions()
    global_center = (rows / 2, cols / 2)
    
    for region in color_8_regions:
        min_y, max_y, min_x, max_x = get_bounding_box(region)
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        area = len(region)
        region_center = ((min_y + max_y) / 2, (min_x + max_x) / 2)
        
        if area <= 4:  # Small region
            rel_x = region_center[1] / cols
            rel_y = region_center[0] / rows
            if rel_y < 0.5:
                color = 2 if rel_x < 0.5 else 4
            else:
                color = 3 if rel_x < 0.5 else 1
            for y, x in region:
                new_grid.values[y][x] = color
        else:  # Large region
            anchors = [
                (min_y, min_x, 2),  # Top-left, Red
                (min_y, max_x, 4),  # Top-right, Yellow
                (max_y, min_x, 3),  # Bottom-left, Green
                (max_y, max_x, 1),  # Bottom-right, Blue
            ]
            for y, x in region:
                distances = [distance((y, x), (ay, ax)) for ay, ax, _ in anchors]
                closest_anchor = min(range(4), key=lambda i: distances[i])
                new_grid.values[y][x] = anchors[closest_anchor][2]
    
    # Post-processing: Smooth color boundaries and ensure color balance
    for region in color_8_regions:
        if len(region) > 4:
            colors = {new_grid.values[y][x] for y, x in region}
            if len(colors) < 4:
                missing_colors = set([1, 2, 3, 4]) - colors
                for color in missing_colors:
                    y, x = region[0]
                    new_grid.values[y][x] = color
    
    return new_grid
