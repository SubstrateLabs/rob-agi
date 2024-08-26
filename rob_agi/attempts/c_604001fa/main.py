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
    3. Sorting blue shapes based on their top-left position and a global counter
    4. Transforming blue shapes into a sequence of colors: green (3), magenta (6), yellow (4), sky (8)
    The color sequence and shape counting are maintained across multiple function calls.
    """
    global blue_shape_counter
    grid = input_grid.deep_copy()
    
    # Identify blue shapes
    blue_regions = grid.find_connected_regions(1)
    
    # Create a list of blue shapes with their top-left positions and a counter
    blue_shapes = []
    for region in blue_regions:
        top_left = min(region)
        blue_shapes.append((top_left, blue_shape_counter, region))
        blue_shape_counter += 1
    
    # Sort blue shapes based on top-left positions and the counter
    blue_shapes.sort(key=lambda x: (x[0], x[1]))
    
    # Transform blue shapes
    for _, _, region in blue_shapes:
        color = get_next_color()
        for r, c in region:
            grid.values[r][c] = color
    
    # Remove all non-transformed colors (set to 0)
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in color_cycle and grid.values[r][c] != 0:
                grid.values[r][c] = 0
    
    return grid
