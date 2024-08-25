from rob_agi.colored_grid import ColoredGrid
from itertools import cycle

# Global color cycle and current color
color_cycle = cycle([3, 6, 4, 8])
current_color = next(color_cycle)

def solve_604001fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Removing all orange (7) shapes
    2. Identifying blue (1) shapes
    3. Sorting blue shapes based on their top-left position
    4. Transforming blue shapes into a sequence of colors: green (3), magenta (6), yellow (4), sky (8)
    The color sequence is maintained across multiple function calls.
    """
    global current_color
    grid = input_grid.deep_copy()
    
    # Remove orange shapes
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 7:
                grid.values[r][c] = 0
    
    # Identify blue shapes
    blue_regions = grid.find_connected_regions(1)
    
    # Sort blue regions
    def region_sort_key(region):
        return min((r, c) for r, c in region)
    
    blue_regions.sort(key=region_sort_key)
    
    # Transform blue shapes
    for region in blue_regions:
        for r, c in region:
            grid.values[r][c] = current_color
        current_color = next(color_cycle)
    
    return grid
