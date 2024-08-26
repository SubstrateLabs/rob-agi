from rob_agi.colored_grid import ColoredGrid

def solve_f3cdc58f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern of colored columns in the bottom-left corner.
    
    The solution follows these steps:
    1. Count the occurrences of colors 1 (blue), 2 (red), 3 (green), and 4 (yellow) in the input grid.
    2. Calculate the maximum possible height for each color column, ensuring a non-decreasing sequence.
    3. Create columns for each color, starting from the bottom-left corner.
    4. The height of each column is based on its color count and the non-decreasing sequence rule.
    5. Columns are arranged in order (1, 2, 3, 4) from left to right.
    6. The rest of the grid is filled with 0 (black/empty).

    This approach creates a stair-step pattern of colored columns while maintaining
    the relative proportions of colors from the input grid and ensuring the non-decreasing height sequence.
    """
    height, width = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    color_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # Count colors
    for row in input_grid.values:
        for color in row:
            if color in color_counts:
                color_counts[color] += 1

    # Calculate maximum possible heights
    max_heights = [0] * 4
    max_heights[3] = min(color_counts[4], height)
    for i in range(2, -1, -1):
        max_heights[i] = min(color_counts[i+1], max_heights[i+1], height)

    # Adjust heights to ensure non-decreasing sequence
    for i in range(3):
        max_heights[i] = min(max_heights[i], max_heights[i+1])

    # Fill the new grid
    for col, color in enumerate([1, 2, 3, 4]):
        for row in range(height - max_heights[col], height):
            new_grid[row][col] = color

    return ColoredGrid(values=new_grid)
