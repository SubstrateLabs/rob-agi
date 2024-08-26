from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Count the occurrences of colors 1, 2, 3, and 4 in the input grid.
    2. Create columns for each color, starting from the bottom-left corner.
    3. The height of each column is the minimum of its color count and the maximum available height.
    4. Columns are arranged in order (1, 2, 3, 4) with non-increasing heights.
    5. The rest of the grid is filled with 0 (black/empty).

    This approach creates a stair-step pattern of colored columns while maintaining
    the relative proportions of colors from the input grid.
    """
    height, width = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # Count colors
    for row in range(height):
        for col in range(width):
            color = input_grid.values[row][col]
            if color in color_counts:
                color_counts[color] += 1

    start_row = height - 1
    start_col = 0

    # Calculate column heights
    max_height = height - start_row
    heights = {}
    for color in [1, 2, 3, 4]:
        heights[color] = min(color_counts[color], max_height)
        max_height = heights[color]

    # Fill the new grid
    for color in [1, 2, 3, 4]:
        for row in range(start_row, start_row - heights[color], -1):
            new_grid[row][start_col] = color
        start_col += 1

    return ColoredGrid(values=new_grid)
