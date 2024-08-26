from rob_agi.colored_grid import ColoredGrid

def solve_695367ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 15x15 grid with the following pattern:
    1. Create a 3x3 grid of small squares (3x3 for 1x1 or 2x2 inputs, 5x5 for 3x3 to 5x5 inputs).
    2. Draw separating lines using the input color at indices based on the small square size.
    3. In each small square, replicate the input pattern, centered for small inputs or top-left aligned for larger inputs.
    4. Fill the rest of the grid with black (0).
    """
    # Create the output grid
    output_grid = [[0 for _ in range(15)] for _ in range(15)]

    # Determine input dimensions and color
    input_height, input_width = input_grid.get_dimensions()
    color = input_grid.values[0][0]

    # Determine small square size and separating line indices
    small_square_size = 5 if max(input_height, input_width) > 3 else 3
    separating_line_indices = [
        small_square_size - 1,
        2 * small_square_size,
        3 * small_square_size + 1
    ]

    # Draw the separating lines
    for i in separating_line_indices:
        for j in range(15):
            output_grid[i][j] = color
            output_grid[j][i] = color

    # Replicate the input pattern
    for block_row in range(3):
        for block_col in range(3):
            start_row = block_row * (small_square_size + 1)
            start_col = block_col * (small_square_size + 1)
            
            if max(input_height, input_width) < small_square_size:
                offset = (small_square_size - max(input_height, input_width)) // 2
            else:
                offset = 0
            
            for input_row in range(input_height):
                for input_col in range(input_width):
                    output_row = start_row + offset + input_row
                    output_col = start_col + offset + input_col
                    if output_row < 15 and output_col < 15 and output_row not in separating_line_indices and output_col not in separating_line_indices:
                        output_grid[output_row][output_col] = input_grid.values[input_row][input_col]

    return ColoredGrid(values=output_grid)
