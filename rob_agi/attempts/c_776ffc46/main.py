
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions with more than one cell change to Red (2)
    2. Red (2) regions with more than one cell change to Green (3)
    3. Green (3) regions with more than one cell change to Blue (1)
    Single-cell regions and other colors remain unchanged.

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Define the color transformation sequence: Blue -> Red -> Green -> Blue.
    3. Find connected regions for each color (1, 2, 3).
    4. Transform regions with more than one cell according to the rules.
    5. Return the transformed grid.
    """
    output_grid = input_grid.deep_copy()
    
    color_transform = {1: 2, 2: 3, 3: 1}
    
    for color in [1, 2, 3]:
        regions = output_grid.find_connected_regions(color)
        for region in regions:
            if len(region) > 1:
                new_color = color_transform[color]
                for r, c in region:
                    output_grid.values[r][c] = new_color
    
    return output_grid
