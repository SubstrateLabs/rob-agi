from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Count the occurrences of colors 1 (blue), 2 (red), 3 (green), and 4 (yellow) in the input grid.
    2. Calculate initial column heights based on color counts and grid height.
    3. Adjust heights to start from the bottom, maintaining relative proportions.
    4. Create columns for each color, starting from the bottom-left corner.
    5. Columns are arranged in order (1, 2, 3, 4) from left to right.
    6. The rest of the grid is filled with 0 (black/empty).

    This approach creates a stair-step pattern of colored columns while maintaining
    the relative proportions of colors from the input grid.
    """
    height, width = input_grid.get_dimensions()
    
    # Step 1: Count colors
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in input_grid.values:
        for color in row:
            if color in color_counts:
                color_counts[color] += 1
    
    # Step 2: Calculate initial column heights
    heights = [0] * 5  # Index 0 won't be used
    heights[4] = min(color_counts[4], height)
    for i in range(3, 0, -1):
        heights[i] = min(color_counts[i], heights[i+1])
    
    # Step 3: Adjust heights to start from bottom
    min_height = min(heights[1:])
    for i in range(1, 5):
        heights[i] = max(0, heights[i] - min_height)
    
    # Step 4: Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    for color in range(1, 5):
        for row in range(height - heights[color], height):
            new_grid[row][color - 1] = color
    
    # Step 5: Return new grid
    return ColoredGrid(values=new_grid)
