from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    y_coords, x_coords = zip(*region)
    return min(y_coords), max(y_coords), min(x_coords), max(x_coords)

def solve_103eff5b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing color 8 regions with a specific color pattern.
    
    The solution works as follows:
    1. Preserve the existing colored pattern.
    2. Identify all connected regions of color 8.
    3. For each color 8 region:
       a. Calculate its bounding box and shape characteristics.
       b. Apply a flexible color mapping based on the region's shape:
          - Red (2) primarily in the top-left
          - Yellow (4) in the top-right and/or bottom-left
          - Blue (1) primarily in the bottom-right
          - Green (3) filling the remaining space, often in the middle or bottom
    4. Transform the grid by applying the color mapping to each color 8 region.
    5. Return the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    color_8_regions = input_grid.find_connected_regions(8)
    
    for region in color_8_regions:
        min_y, max_y, min_x, max_x = get_bounding_box(region)
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        area = len(region)
        aspect_ratio = width / height

        def color_map(rel_x: float, rel_y: float) -> int:
            if area <= 4:  # Special case for very small regions
                return [2, 4, 1, 3][len(region) - 1]
            
            if aspect_ratio > 2 or aspect_ratio < 0.5:  # Long, thin regions
                if width > height:
                    return 2 if rel_x < 0.25 else 4 if rel_x < 0.5 else 1 if rel_x < 0.75 else 3
                else:
                    return 2 if rel_y < 0.25 else 4 if rel_y < 0.5 else 1 if rel_y < 0.75 else 3
            
            # General case
            if rel_x + rel_y < 0.8:
                return 2 if rel_x < rel_y else 4
            elif rel_x + rel_y > 1.2:
                return 1 if rel_x > rel_y else 3
            else:
                return 4 if rel_x < 0.5 else 1

        for y, x in region:
            rel_x = (x - min_x) / (width - 1)
            rel_y = (y - min_y) / (height - 1)
            new_grid.values[y][x] = color_map(rel_x, rel_y)
    
    return new_grid
