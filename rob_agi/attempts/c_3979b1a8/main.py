from rob_agi.colored_grid import ColoredGrid

def solve_3979b1a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 input grid into a 10x10 output grid by:
    1. Copying the input to the top-left quadrant
    2. Creating a vertical stripe pattern on the right side
    3. Creating a horizontal stripe pattern on the bottom
    4. Filling the bottom-right quadrant with a specific pattern
    
    The pattern uses the corner color, center color, and a special color from the input grid.
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

    # Fill the right side with vertical stripes (columns 5-9)
    for i in range(10):
        new_grid[i][5:] = [corner_color, center_color, special_color, corner_color, center_color]

    # Fill the bottom with horizontal stripes (rows 5-9)
    for i in range(5, 10):
        new_grid[i][:5] = [corner_color, center_color, special_color, corner_color, center_color]

    # Fill the bottom-right quadrant (rows 5-9, columns 5-9)
    for i in range(5, 10):
        for j in range(5, 10):
            if i == j:
                new_grid[i][j] = special_color
            elif i > j:
                new_grid[i][j] = center_color
            else:
                new_grid[i][j] = corner_color

    return ColoredGrid(values=new_grid)
