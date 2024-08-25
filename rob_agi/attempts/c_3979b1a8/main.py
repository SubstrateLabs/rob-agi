from rob_agi.colored_grid import ColoredGrid

def solve_3979b1a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 input grid into a 10x10 output grid by:
    1. Copying the input to the top-left quadrant
    2. Expanding the pattern to fill the 10x10 space using corner and center colors
    3. Creating specific patterns for each quadrant
    4. Setting the bottom-right corner to the color of input[1][1]

    The top-right and bottom-left quadrants follow a specific pattern using corner and center colors.
    The bottom-right quadrant has its own pattern, with the last cell being a special case.
    """
    # Extract key information
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
        new_grid[i][5] = corner_color
        new_grid[i][6] = center_color
        new_grid[i][7] = input_grid.values[i][1]
        new_grid[i][8] = corner_color
        new_grid[i][9] = center_color

    # Fill the bottom-left quadrant (rows 5-9, columns 0-4)
    for j in range(5):
        new_grid[5][j] = corner_color
        new_grid[6][j] = center_color
        new_grid[7][j] = input_grid.values[1][j]
        new_grid[8][j] = corner_color
        new_grid[9][j] = center_color

    # Fill the bottom-right quadrant (rows 5-9, columns 5-9)
    for i in range(5, 10):
        for j in range(5, 10):
            if i == 5 or i == 8:
                new_grid[i][j] = corner_color
            elif i == 6 or i == 9:
                new_grid[i][j] = center_color
            elif i == 7:
                if j in [5, 8]:
                    new_grid[i][j] = corner_color
                elif j == 7:
                    new_grid[i][j] = special_color
                else:
                    new_grid[i][j] = center_color

    # Set the bottom-right corner to the special color
    new_grid[9][9] = special_color

    return ColoredGrid(values=new_grid)
