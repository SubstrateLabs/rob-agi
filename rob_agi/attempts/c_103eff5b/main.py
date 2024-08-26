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
       a. Find its bounding box.
       b. Divide the region into sections:
          - Top third: color 2 (red)
          - Bottom third: color 3 (green)
          - Middle section: left half color 4 (yellow), right half color 1 (blue)
    4. Transform the grid by applying the color mapping to each color 8 region.
    5. Return the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    color_8_regions = input_grid.find_connected_regions(8)
    
    for region in color_8_regions:
        min_y, max_y, min_x, max_x = get_bounding_box(region)
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        top_height = (height + 2) // 3  # Round up
        bottom_height = (height + 2) // 3  # Round up
        middle_height = height - top_height - bottom_height
        
        for y, x in region:
            if y - min_y < top_height:
                new_color = 2  # Top: red
            elif y - min_y >= height - bottom_height:
                new_color = 3  # Bottom: green
            else:
                if x - min_x < width // 2:
                    new_color = 4  # Middle-left: yellow
                else:
                    new_color = 1  # Middle-right: blue
            
            new_grid.values[y][x] = new_color
    
    return new_grid
