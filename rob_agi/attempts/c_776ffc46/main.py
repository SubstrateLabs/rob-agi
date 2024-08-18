
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    
    color_transform = {1: 2, 2: 3, 3: 1}
    
    for color in [1, 2, 3]:
        regions = output_grid.find_connected_regions(color)
        for region in regions:
            if len(region) > 1:  # Only transform regions with more than one cell
                new_color = color_transform[color]
                for r, c in region:
                    output_grid.values[r][c] = new_color
    
    # Handle single-cell regions
    for r in range(len(output_grid.values)):
        for c in range(len(output_grid.values[0])):
            if output_grid.values[r][c] in color_transform:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                if all(not (0 <= nr < len(output_grid.values) and 
                            0 <= nc < len(output_grid.values[0]) and 
                            output_grid.values[nr][nc] == output_grid.values[r][c])
                       for nr, nc in neighbors):
                    # This is a single-cell region, don't transform it
                    pass
                else:
                    # This cell is part of a larger region, transform it
                    output_grid.values[r][c] = color_transform[output_grid.values[r][c]]
    
    return output_grid
