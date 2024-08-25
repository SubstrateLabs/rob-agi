from rob_agi.colored_grid import ColoredGrid

def solve_c1990cce(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a single-row input grid into a square grid with a diamond pattern.
    
    The function creates a pattern as follows:
    1. Copy the input row to the first row of the output grid.
    2. Create a red diamond by expanding diagonally from the center red square to all sides.
    3. Fill in a blue checkerboard pattern within the red diamond, starting from the row
       below the top point of the diamond.
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
        diamond_width = row * 2 + 1
        left = max(0, center - row)
        right = min(grid_size - 1, center + row)
        output_grid.values[row][left] = 2
        output_grid.values[row][right] = 2
    
    # Step 4: Create the blue checkerboard pattern within the red diamond
    for row in range(1, grid_size):
        diamond_width = min(row * 2 + 1, (grid_size - row) * 2 - 1)
        left = max(0, center - diamond_width // 2)
        right = min(grid_size - 1, center + diamond_width // 2)
        for col in range(left + 1, right):
            if (row + col) % 2 == 1 and output_grid.values[row][col] == 0:
                output_grid.values[row][col] = 1
    
    return output_grid
