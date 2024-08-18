
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    
    color_transform = {1: 2, 2: 3, 3: 1}
    
    # Process colors in order: Blue (1), Red (2), Green (3)
    for color in [1, 2, 3]:
        regions = output_grid.find_connected_regions(color)
        for region in regions:
            if len(region) > 1:  # Only transform regions with more than one cell
                new_color = color_transform[color]
                for r, c in region:
                    output_grid.values[r][c] = new_color
    
    return output_grid
