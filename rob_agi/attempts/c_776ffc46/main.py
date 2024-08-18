
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    blue_regions = output_grid.find_connected_regions(1)  # Find all blue regions
    
    for region in blue_regions:
        for r, c in region:
            output_grid.values[r][c] = 2  # Change blue (1) to red (2)
    
    return output_grid
