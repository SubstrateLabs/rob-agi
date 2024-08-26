import math
from rob_agi.colored_grid import ColoredGrid

def solve_762cd429(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding a 2x2 color pattern from the bottom-left corner
    into a larger pattern that fills the right two-thirds and bottom half of the grid.
    
    1. Extracts the 2x2 color pattern from the bottom-left corner.
    2. Calculates the expansion size based on the grid dimensions.
    3. Creates a new grid and fills it with the expanded pattern:
       - Bottom half uses a solid color if bottom colors match, otherwise a checkerboard.
       - Top half uses a checkerboard pattern of the top colors.
    4. Copies the unchanged area from the input grid to the new grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    input_values = input_grid.values
    height, width = len(input_values), len(input_values[0])
    
    # Extract 2x2 color pattern
    top_left = input_values[height-2][0]
    top_right = input_values[height-2][1]
    bottom_left = input_values[height-1][0]
    bottom_right = input_values[height-1][1]
    
    # Calculate expansion size
    expansion_width = math.ceil(width * 2/3)
    expansion_height = math.ceil(height * 1/2)
    
    # Calculate checkerboard square size
    square_width = math.ceil(expansion_width / 2)
    square_height = math.ceil(expansion_height / 2)
    
    # Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill expanded pattern
    for y in range(height - expansion_height, height):
        for x in range(width - expansion_width, width):
            if y < height - expansion_height // 2:  # Top half
                if ((x - (width - expansion_width)) // square_width + (y - (height - expansion_height)) // square_height) % 2 == 0:
                    new_grid[y][x] = top_left
                else:
                    new_grid[y][x] = top_right
            else:  # Bottom half
                if bottom_left == bottom_right:
                    new_grid[y][x] = bottom_left
                else:
                    if ((x - (width - expansion_width)) // square_width + (y - (height - expansion_height)) // square_height) % 2 == 0:
                        new_grid[y][x] = bottom_left
                    else:
                        new_grid[y][x] = bottom_right
    
    # Copy unchanged area
    for y in range(height):
        for x in range(width):
            if x < width - expansion_width or y < height - expansion_height:
                new_grid[y][x] = input_values[y][x]
    
    return ColoredGrid(values=new_grid)
