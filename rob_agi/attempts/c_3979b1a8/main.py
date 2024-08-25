from rob_agi.colored_grid import ColoredGrid

def solve_3979b1a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 input grid into a 10x10 output grid by:
    1. Copying the input to the top-left quadrant
    2. Mirroring and expanding the pattern to fill the 10x10 space
    3. Using the center color of the input to fill specific rows and columns
    
    The right half copies columns 0, center color, 1, 0, center color.
    The bottom half copies rows 0, center color, 1, 0, center color.
    """
    # Extract important information
    input_size = input_grid.get_dimensions()[0]  # Assuming square input
    center_color = input_grid.values[input_size // 2][input_size // 2]

    # Create a new 10x10 grid
    new_grid = [[0 for _ in range(10)] for _ in range(10)]

    # Copy the input 5x5 grid to the top-left quadrant
    for i in range(input_size):
        for j in range(input_size):
            new_grid[i][j] = input_grid.values[i][j]

    # Fill the right half (columns 5-9)
    for i in range(input_size):
        new_grid[i][5] = new_grid[i][0]  # Copy column 0 to column 5
        new_grid[i][6] = center_color    # Fill column 6 with center color
        new_grid[i][7] = new_grid[i][1]  # Copy column 1 to column 7
        new_grid[i][8] = new_grid[i][0]  # Copy column 0 to column 8
        new_grid[i][9] = center_color    # Fill column 9 with center color

    # Fill the bottom half (rows 5-9)
    for j in range(10):
        new_grid[5][j] = new_grid[0][j]  # Copy row 0 to row 5
        new_grid[6][j] = center_color    # Fill row 6 with center color
        new_grid[7][j] = new_grid[1][j]  # Copy row 1 to row 7
        new_grid[8][j] = new_grid[0][j]  # Copy row 0 to row 8
        new_grid[9][j] = center_color    # Fill row 9 with center color

    return ColoredGrid(values=new_grid)
