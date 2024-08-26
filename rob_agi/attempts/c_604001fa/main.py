from rob_agi.colored_grid import ColoredGrid

# Global variables
color_cycle = [3, 6, 4, 8]
blue_shape_counter = 0
color_index = 0

def get_next_color():
    global color_index
    color = color_cycle[color_index]
    color_index = (color_index + 1) % len(color_cycle)
    return color

def solve_604001fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Ignoring all orange (7) shapes
    2. Identifying blue (1) shapes
    3. Transforming all blue shapes in a single grid to the same color
    4. Cycling through colors (green (3), magenta (6), yellow (4), sky (8)) for each new input grid
    The color sequence is maintained across multiple function calls.
    """
    global blue_shape_counter
    grid = input_grid.deep_copy()
    
    # Identify blue shapes
    blue_regions = grid.find_connected_regions(1)
    
    if blue_regions:
        # Get the next color for all blue shapes in this grid
        color = get_next_color()
        
        # Transform all blue shapes to the same color
        for region in blue_regions:
            for r, c in region:
                grid.values[r][c] = color
    
    # Remove all non-transformed colors (set to 0)
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in color_cycle and grid.values[r][c] != 0:
                grid.values[r][c] = 0
    
    return grid
