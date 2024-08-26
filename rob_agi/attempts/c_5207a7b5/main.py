from rob_agi.colored_grid import ColoredGrid

def solve_5207a7b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid according to the following rules:
    1. Find the gray vertical line (color 5) in the input grid.
    2. Create a sky blue (color 8) triangle on the left side, with base width equal to the gray line's column index.
       The triangle decreases in width by 1 every two rows, continuing as a single column to the bottom of the grid.
    3. Preserve the gray line in its original position and length.
    4. Add a magenta (color 6) shape to the right of the gray line, starting with width 1 and increasing by 1 each row,
       up to a maximum width of 3 or the available space.
    5. Leave all other cells black (color 0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the rules.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    # Find gray line column and length
    gray_line_column = next(col for col in range(width) if input_grid.values[0][col] == 5)
    gray_line_length = sum(1 for row in input_grid.values if row[gray_line_column] == 5)

    # Draw sky blue triangle and column
    for row in range(height):
        sky_blue_width = max(1, gray_line_column - row // 2)
        for col in range(min(sky_blue_width, gray_line_column)):
            new_grid.values[row][col] = 8

    # Preserve gray line
    for row in range(gray_line_length):
        new_grid.values[row][gray_line_column] = 5

    # Add magenta shape
    for row in range(gray_line_length):
        magenta_width = min(row + 1, 3, width - gray_line_column - 1)
        for col in range(gray_line_column + 1, gray_line_column + 1 + magenta_width):
            new_grid.values[row][col] = 6

    return new_grid
