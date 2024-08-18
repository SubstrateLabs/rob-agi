
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
    3. Iterate multiple times to handle cascading effects.
    4. In each iteration, process colors in the specific order: Blue, Red, Green.
    5. Find connected regions for each color and transform them if needed.
    6. Continue iterations until no more changes occur.
    7. Return the transformed grid.
    """
    output_grid = input_grid.deep_copy()
    color_transform = {1: 2, 2: 3, 3: 1}
    colors = [1, 2, 3]  # Blue, Red, Green
    
    changes_made = True
    while changes_made:
        changes_made = False
        for color in colors:
            regions = output_grid.find_connected_regions(color)
            for region in regions:
                if len(region) > 1:
                    new_color = color_transform[color]
                    for r, c in region:
                        output_grid.values[r][c] = new_color
                        changes_made = True
        
        # Apply changes immediately after processing all colors
        output_grid = output_grid.deep_copy()
    
    return output_grid
