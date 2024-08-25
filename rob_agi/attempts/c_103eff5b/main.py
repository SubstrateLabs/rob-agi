from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    x_coords, y_coords = zip(*region)
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def solve_103eff5b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing color 8 regions with a quadrant-based color pattern.
    
    The solution works as follows:
    1. Preserve the small colored pattern in the top-left corner.
    2. Identify all connected regions of color 8.
    3. For each color 8 region:
       a. Find its bounding box.
       b. Divide the bounding box into four quadrants.
       c. Assign colors to the quadrants:
          - Top-left: color 2 (red)
          - Top-right: color 4 (yellow)
          - Bottom-left: color 3 (green)
          - Bottom-right: color 1 (blue)
    4. Transform the grid by applying the quadrant-based color mapping to each color 8 region.
    5. Return the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    color_8_regions = input_grid.find_connected_regions(8)
    
    for region in color_8_regions:
        min_x, max_x, min_y, max_y = get_bounding_box(region)
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if input_grid.values[y][x] == 8:  # Only change color 8 cells
                    if x <= mid_x:
                        if y <= mid_y:
                            new_color = 2  # Top-left: red
                        else:
                            new_color = 3  # Bottom-left: green
                    else:
                        if y <= mid_y:
                            new_color = 4  # Top-right: yellow
                        else:
                            new_color = 1  # Bottom-right: blue
                    
                    new_grid.values[y][x] = new_color
    
    return new_grid
