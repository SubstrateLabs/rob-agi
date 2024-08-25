from rob_agi.colored_grid import ColoredGrid

def solve_c1990cce(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a single-row input grid into a square grid with a diamond pattern.
    
    The function creates a red diamond shape starting from the red square in the input,
    then fills in an inverted blue triangle below the diamond. The pattern is as follows:
    1. Copy the input row to the first row of the output grid.
    2. Create a red diamond by expanding diagonally from the center red square.
    3. Fill in blue squares below the red diamond, starting from the 4th row,
       avoiding placing blue squares directly under red ones.
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
    for row in range(1, grid_size):
        left = center - row
        right = center + row
        if 0 <= left < grid_size:
            output_grid.values[row][left] = 2
        if 0 <= right < grid_size:
            output_grid.values[row][right] = 2
    
    # Step 4: Create the blue inverted triangle
    for row in range(3, grid_size):
        left_red = output_grid.values[row].index(2)
        right_red = grid_size - 1 - output_grid.values[row][::-1].index(2)
        for col in range(left_red + 1, right_red):
            if output_grid.values[row-1][col] != 2:
                output_grid.values[row][col] = 1
    
    # Step 5: Return the completed grid
    return output_grid
