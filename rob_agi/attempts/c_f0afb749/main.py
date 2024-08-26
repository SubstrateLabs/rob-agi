from rob_agi.colored_grid import ColoredGrid

def solve_f0afb749(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Doubling the grid size
    2. Expanding non-black squares into 2x2 blocks
    3. Adding blue squares (1s) diagonally from top-left to bottom-right
       in the entire grid, starting from the second row and second column
    4. Preserving the expanded non-black squares when adding the blue diagonal
    """
    # Step 1: Initialize the output grid
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    # Step 2: Expand non-black squares
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                output_grid[r*2][c*2] = color
                output_grid[r*2][c*2+1] = color
                output_grid[r*2+1][c*2] = color
                output_grid[r*2+1][c*2+1] = color

    # Step 3: Add blue diagonal
    for i in range(1, output_rows):
        j = i - 1
        if j < output_cols and output_grid[i][j] == 0:
            output_grid[i][j] = 1  # Set to blue

    # Step 4: Create and return the final ColoredGrid
    return ColoredGrid(values=output_grid)
