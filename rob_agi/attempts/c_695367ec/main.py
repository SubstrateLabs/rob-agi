from rob_agi.colored_grid import ColoredGrid

def solve_695367ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 15x15 grid with the following pattern:
    1. Create a 3x3 grid of 5x5 squares.
    2. Draw separating lines using the input color at indices 4, 9, and 14 (both horizontally and vertically).
    3. In each 5x5 square, replicate the input pattern, centered and padded with black (0) if necessary.
    4. Fill the rest of the grid with black (0).
    """
    # Create the output grid
    output_grid = [[0 for _ in range(15)] for _ in range(15)]

    # Determine input dimensions
    input_height, input_width = input_grid.get_dimensions()

    # Calculate padding
    vertical_padding = (5 - input_height) // 2
    horizontal_padding = (5 - input_width) // 2

    # Get the color from the input grid
    color = input_grid.values[0][0]

    # Draw the separating lines
    for i in [4, 9, 14]:
        for j in range(15):
            output_grid[i][j] = color
            output_grid[j][i] = color

    # Replicate the input pattern
    for block_row in range(3):
        for block_col in range(3):
            start_row = block_row * 5
            start_col = block_col * 5
            for input_row in range(input_height):
                for input_col in range(input_width):
                    output_row = start_row + vertical_padding + input_row
                    output_col = start_col + horizontal_padding + input_col
                    if output_row % 5 != 4 and output_col % 5 != 4:
                        output_grid[output_row][output_col] = input_grid.values[input_row][input_col]

    return ColoredGrid(values=output_grid)
