from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    x_coords, y_coords = zip(*region)
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def color_mapping(point: Tuple[int, int], bounding_box: Tuple[int, int, int, int]) -> int:
    x, y = point
    x_min, x_max, y_min, y_max = bounding_box
    x_mid, y_mid = (x_min + x_max) // 2, (y_min + y_max) // 2
    
    if x <= x_mid:
        return 2 if y <= y_mid else 3
    else:
        return 4 if y <= y_mid else 1

def solve_103eff5b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing color 8 regions with a quadrant-based color pattern.
    
    The solution works as follows:
    1. Preserve the small colored pattern in the top-left corner.
    2. Identify all connected regions of color 8.
    3. For each color 8 region:
       a. Find its bounding box.
       b. Divide the bounding box into four quadrants.
       c. Assign colors to the quadrants in clockwise order:
          - Top-left: color 2 (red)
          - Top-right: color 4 (yellow)
          - Bottom-right: color 1 (blue)
          - Bottom-left: color 3 (green)
    4. Transform the grid by applying the quadrant-based color mapping to each color 8 region.
    5. Return the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    color_8_regions = input_grid.find_connected_regions(8)
    
    for region in color_8_regions:
        bounding_box = get_bounding_box(region)
        new_grid = new_grid.apply_function_to_regions(
            lambda r: color_mapping((r[0], r[1]), bounding_box),
            [region]
        )
    
    return new_grid
