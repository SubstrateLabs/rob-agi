from rob_agi.colored_grid import ColoredGrid

def solve_c1990cce(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a single-row input grid into a square grid with a diamond pattern.
    
    The function creates a red diamond shape starting from the red square in the input,
    then fills in a blue diamond pattern in the bottom half of the grid. The pattern is as follows:
    1. Copy the input row to the first row of the output grid.
    2. Create a red diamond by expanding diagonally from the center red square to all sides.
    3. Fill in blue squares in the bottom half of the grid, creating a complementary diamond pattern.
       Blue squares are placed in alternating positions, creating diagonal lines.
    4. The rest of the grid remains black (empty space).

    Args:
    input_grid (ColoredGrid): A single-row grid with one red square (2) and the rest black (0).

    Returns:
    ColoredGrid: A square grid with the described pattern.
    """
    # Step 1: Initialize the output grid
    grid_size = len(input_grid.values[0])
    output_grid = ColoredGrid(values=[[0 for _ in range(grid_size)] for _ in range(grid_size)])
    
    # Step 2: Copy the input row and find center
    output_grid.values[0] = input_grid.values[0].copy()
    center = input_grid.values[0].index(2)
    
    # Step 3: Create the red diamond
    for row in range(grid_size):
        left = center - row
        right = center + row
        if 0 <= left < grid_size:
            output_grid.values[row][left] = 2
        if 0 <= right < grid_size:
            output_grid.values[row][right] = 2
    
    # Step 4: Create the blue diamond pattern
    middle = grid_size // 2
    for row in range(middle, grid_size):
        for col in range(grid_size):
            if output_grid.values[row][col] == 0 and (row + col) % 2 == 1:
                output_grid.values[row][col] = 1
    
    return output_grid
