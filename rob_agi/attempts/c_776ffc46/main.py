
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    color_transform = {1: 2, 2: 3, 3: 1}
    
    for r in range(rows):
        for c in range(cols):
            current_color = output_grid.values[r][c]
            if current_color in color_transform:
                # Only transform colors in connected regions
                region = output_grid.find_connected_regions(current_color)
                if region:
                    for region_r, region_c in region[0]:
                        output_grid.values[region_r][region_c] = color_transform[current_color]
    
    return output_grid
