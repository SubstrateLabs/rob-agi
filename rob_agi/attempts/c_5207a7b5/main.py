from rob_agi.colored_grid import ColoredGrid

def solve_5207a7b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid according to the following rules:
    1. Find the gray vertical line (color 5) in the input grid.
    2. Create a sky blue (color 8) shape on the left side:
       - Start with a width equal to the gray line's column index.
       - Maintain this width for a number of rows equal to the gray line's length.
       - After that, decrease the width by 1 every two rows.
       - Continue as a single column until the total height is 2 * (gray line column index + 1).
    3. Preserve the gray line in its original position and length.
    4. Add a magenta (color 6) shape to the right of the gray line:
       - Width is calculated as: 3 - (gray line column index % 3)
       - Height is equal to the gray line's length.
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
    sky_blue_height = min(2 * (gray_line_column + 1), height)
    for row in range(sky_blue_height):
        if row < gray_line_length:
            sky_blue_width = gray_line_column
        else:
            sky_blue_width = max(1, gray_line_column - (row - gray_line_length + 1) // 2)
        for col in range(sky_blue_width):
            new_grid.values[row][col] = 8

    # Preserve gray line
    for row in range(gray_line_length):
        new_grid.values[row][gray_line_column] = 5

    # Add magenta shape
    magenta_width = 3 - (gray_line_column % 3)
    for row in range(gray_line_length):
        for col in range(gray_line_column + 1, min(gray_line_column + 1 + magenta_width, width)):
            new_grid.values[row][col] = 6

    return new_grid
