from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_516b51b7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a concentric pattern to connected blue regions.
    
    The solution follows these steps:
    1. Find all connected blue (1) regions in the input grid.
    2. For each region:
       a. Determine its dimensions and the number of layers based on the smaller dimension.
       b. Apply concentric layers of colors from outside to inside:
          - Outermost layer: Blue (1)
          - Alternating layers: Red (2) and Green (3)
       c. Handle the center based on remaining space:
          - Fill with the last layer color if only one cell remains in any dimension
          - Fill with Red (2) if a 2x2 or larger area remains
    3. Return the transformed grid.
    
    This pattern is applied consistently to all blue regions, regardless of their size or position.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    def get_region_dimensions(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1
    
    def get_layer_count(smaller_dimension: int) -> int:
        if smaller_dimension <= 3:
            return 2
        elif smaller_dimension <= 5:
            return 3
        elif smaller_dimension <= 7:
            return 4
        else:
            return 5
    
    def get_layer_color(layer_index: int) -> int:
        if layer_index == 0:
            return 1  # Blue
        return 2 if layer_index % 2 == 1 else 3  # Red or Green
    
    def color_region(region: List[Tuple[int, int]]):
        min_r, min_c, height, width = get_region_dimensions(region)
        smaller_dim = min(height, width)
        layer_count = get_layer_count(smaller_dim)
        
        for layer in range(layer_count):
            color = get_layer_color(layer)
            for r in range(min_r + layer, min_r + height - layer):
                if layer < width // 2:
                    output_grid.values[r][min_c + layer] = color
                    output_grid.values[r][min_c + width - 1 - layer] = color
            for c in range(min_c + layer, min_c + width - layer):
                if layer < height // 2:
                    output_grid.values[min_r + layer][c] = color
                    output_grid.values[min_r + height - 1 - layer][c] = color
        
        # Handle center
        center_height = height - 2 * (layer_count - 1)
        center_width = width - 2 * (layer_count - 1)
        center_color = 2 if center_height > 1 and center_width > 1 else get_layer_color(layer_count - 1)
        for r in range(min_r + layer_count - 1, min_r + height - layer_count + 1):
            for c in range(min_c + layer_count - 1, min_c + width - layer_count + 1):
                output_grid.values[r][c] = center_color
    
    blue_regions = input_grid.find_connected_regions(1)
    for region in blue_regions:
        color_region(region)
    
    return output_grid
