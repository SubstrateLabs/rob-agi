from rob_agi.colored_grid import ColoredGrid

def solve_5207a7b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid according to the following rules:
    1. Find the gray vertical line (color 5) in the input grid.
    2. Create a sky blue (color 8) shape on the left side:
       - Start with a width equal to the gray line's column index.
       - Maintain this width for a number of rows equal to the gray line's length.
       - After that, decrease the width by 1 every two rows.
       - Continue until the shape narrows to a single column or reaches the bottom of the grid.
    3. Preserve the gray line in its original position and length.
    4. Add a magenta (color 6) shape to the right of the gray line:
       - Initial width is min(3, remaining columns after the gray line).
       - Height is equal to the gray line's length.
       - Decrease width by 1 every two rows, but only within the height of the gray line.
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

    # Draw sky blue shape
    sky_blue_width = gray_line_column
    row = 0
    while row < height and sky_blue_width > 0:
        for col in range(sky_blue_width):
            new_grid.values[row][col] = 8
        if row >= gray_line_length - 1:
            if row % 2 == 1:
                sky_blue_width = max(1, sky_blue_width - 1)
        row += 1

    # Preserve gray line
    for row in range(gray_line_length):
        new_grid.values[row][gray_line_column] = 5

    # Add magenta shape
    magenta_width = min(3, width - gray_line_column - 1)
    for row in range(gray_line_length):
        for col in range(1, magenta_width + 1):
            if gray_line_column + col < width:
                new_grid.values[row][gray_line_column + col] = 6
        if row % 2 == 1 and row < gray_line_length - 1:
            magenta_width = max(0, magenta_width - 1)

    return new_grid
