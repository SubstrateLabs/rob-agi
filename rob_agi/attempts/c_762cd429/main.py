import math
from rob_agi.colored_grid import ColoredGrid

def solve_762cd429(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding a 2x2 color cluster in the bottom-right corner
    into a larger pattern that fills most of the grid.
    
    1. Extracts the 2x2 color cluster from the bottom-right corner.
    2. Calculates the expansion size based on the grid dimensions.
    3. Determines the sizes for each quadrant of the expanded pattern.
    4. Creates a new grid and fills it with the expanded pattern.
    5. Copies the unchanged area from the input grid to the new grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    input_values = input_grid.values
    height, width = len(input_values), len(input_values[0])
    
    # Extract 2x2 color cluster
    top_left = input_values[height-2][width-2]
    top_right = input_values[height-2][width-1]
    bottom_left = input_values[height-1][width-2]
    bottom_right = input_values[height-1][width-1]
    
    # Calculate expansion size
    width_expansion = math.ceil(width / 2)
    height_expansion = math.ceil(height / 2)
    
    # Calculate quadrant sizes
    top_left_height = height_expansion // 2
    top_left_width = width_expansion // 2
    top_right_height = height_expansion // 2
    top_right_width = width_expansion - (width_expansion // 2)
    bottom_left_height = height_expansion - (height_expansion // 2)
    bottom_left_width = width_expansion // 2
    bottom_right_height = height_expansion - (height_expansion // 2)
    bottom_right_width = width_expansion - (width_expansion // 2)
    
    # Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill expanded pattern
    for i in range(height - bottom_right_height, height):
        for j in range(width - bottom_right_width, width):
            new_grid[i][j] = bottom_right

    for i in range(height - bottom_left_height - top_left_height, height - top_left_height):
        for j in range(width - bottom_left_width - bottom_right_width, width - bottom_right_width):
            new_grid[i][j] = bottom_left

    for i in range(height - top_right_height - bottom_right_height, height - bottom_right_height):
        for j in range(width - top_right_width, width):
            new_grid[i][j] = top_right

    for i in range(height - height_expansion, height - top_right_height):
        for j in range(width - width_expansion, width - top_right_width):
            new_grid[i][j] = top_left
    
    # Copy unchanged area
    for i in range(height - height_expansion):
        for j in range(width - width_expansion):
            new_grid[i][j] = input_values[i][j]
    
    return ColoredGrid(values=new_grid)
