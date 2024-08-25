from rob_agi.colored_grid import ColoredGrid

def solve_3979b1a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 input grid into a 10x10 output grid by:
    1. Copying the input to the top-left quadrant
    2. Mirroring the pattern to the right and bottom with specific color changes
    3. Creating a unique pattern in the bottom-right quadrant
    4. Using the corner, center, and a special color from the input grid

    The top-right and bottom-left quadrants mirror the input with color substitutions.
    The bottom-right quadrant has a specific pattern based on the key colors.
    """
    # Extract key colors
    corner_color = input_grid.values[0][0]
    center_color = input_grid.values[2][2]
    special_color = input_grid.values[1][1]

    # Create a new 10x10 grid
    new_grid = [[0 for _ in range(10)] for _ in range(10)]

    # Copy the original 5x5 input to the top-left quadrant
    for i in range(5):
        for j in range(5):
            new_grid[i][j] = input_grid.values[i][j]

    # Fill the top-right quadrant (rows 0-4, columns 5-9)
    for i in range(5):
        new_grid[i][5:] = [corner_color, center_color, input_grid.values[i][1], corner_color, center_color]

    # Fill the bottom-left quadrant (rows 5-9, columns 0-4)
    for j in range(5):
        for i in range(5, 10):
            new_grid[i][j] = [corner_color, center_color, input_grid.values[1][j], corner_color, center_color][i-5]

    # Fill the bottom-right quadrant (rows 5-9, columns 5-9)
    for i in range(5, 10):
        for j in range(5, 10):
            if i == 5 or j == 5:
                new_grid[i][j] = corner_color
            elif i == 6:
                new_grid[i][j] = center_color
            elif i == 7:
                new_grid[i][j] = input_grid.values[1][j-5]
            elif i == 8:
                new_grid[i][j] = corner_color if j < 9 else center_color
            else:  # i == 9
                new_grid[i][j] = center_color if j < 9 else special_color

    return ColoredGrid(values=new_grid)
