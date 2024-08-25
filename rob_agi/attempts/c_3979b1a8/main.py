from rob_agi.colored_grid import ColoredGrid

def solve_3979b1a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 input grid into a 10x10 output grid by:
    1. Copying the input to the top-left quadrant
    2. Expanding the pattern to fill the 10x10 space
    3. Using the corner and center colors to create specific patterns
    
    The right half copies columns 4, center color, 1, 0, center color.
    The bottom half copies rows 4, center color, 1, 0, center color.
    The bottom-right quadrant follows both patterns, with row pattern taking precedence.
    """
    # Extract important information
    input_size = input_grid.get_dimensions()[0]  # Assuming square input
    center_color = input_grid.values[input_size // 2][input_size // 2]
    corner_color = input_grid.values[0][0]

    # Create a new 10x10 grid
    new_grid = [[0 for _ in range(10)] for _ in range(10)]

    # Copy the input 5x5 grid to the top-left quadrant
    for i in range(input_size):
        for j in range(input_size):
            new_grid[i][j] = input_grid.values[i][j]

    # Fill the top-right quadrant (columns 5-9)
    for i in range(input_size):
        new_grid[i][5] = corner_color    # Copy corner color to column 5
        new_grid[i][6] = center_color    # Fill column 6 with center color
        new_grid[i][7] = new_grid[i][1]  # Copy column 1 to column 7
        new_grid[i][8] = corner_color    # Copy corner color to column 8
        new_grid[i][9] = center_color    # Fill column 9 with center color

    # Fill the bottom-left quadrant (rows 5-9)
    for j in range(input_size):
        new_grid[5][j] = corner_color    # Copy corner color to row 5
        new_grid[6][j] = center_color    # Fill row 6 with center color
        new_grid[7][j] = new_grid[1][j]  # Copy row 1 to row 7
        new_grid[8][j] = corner_color    # Copy corner color to row 8
        new_grid[9][j] = center_color    # Fill row 9 with center color

    # Fill the bottom-right quadrant (rows 5-9, columns 5-9)
    for i in range(5, 10):
        for j in range(5, 10):
            if i == 5 or i == 8:
                new_grid[i][j] = corner_color
            elif i == 6 or i == 9:
                new_grid[i][j] = center_color
            elif i == 7:
                if j == 6:
                    new_grid[i][j] = new_grid[1][1]
                elif j == 7:
                    new_grid[i][j] = new_grid[1][1]
                else:
                    new_grid[i][j] = corner_color

    return ColoredGrid(values=new_grid)
