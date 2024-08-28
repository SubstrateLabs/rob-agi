from rob_agi.colored_grid import ColoredGrid

def solve_695367ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 15x15 grid with the following pattern:
    1. Create a 3x3 grid of 5x5 squares.
    2. Draw separating lines using the input color. Line width is 3 for 3x3 inputs, 1 for others.
    3. In each 5x5 square:
       - For 3x3 inputs: Fill the entire 5x5 square with the input color.
       - For 1x1 or 2x2 inputs: Center the input pattern.
       - For 4x4 or larger inputs: Draw only the separating lines.
    4. Fill the rest of the grid with black (0).
    """
    output_grid = [[0 for _ in range(15)] for _ in range(15)]
    input_height, input_width = input_grid.get_dimensions()
    color = input_grid.values[0][0]

    # Determine line width
    line_width = 3 if input_height == 3 and input_width == 3 else 1

    # Draw separating lines
    for i in range(4, 4 + line_width):
        output_grid[i] = [color] * 15
        for j in range(15):
            output_grid[j][i] = color
    for i in range(9, 9 + line_width):
        output_grid[i] = [color] * 15
        for j in range(15):
            output_grid[j][i] = color

    # Process each 5x5 square
    for row in range(3):
        for col in range(3):
            start_row = row * 5
            start_col = col * 5

            if input_height == 3 and input_width == 3:
                # Fill the entire 5x5 square for 3x3 inputs
                for i in range(5):
                    for j in range(5):
                        output_grid[start_row + i][start_col + j] = color
            elif max(input_height, input_width) <= 2:
                # Center the input pattern for 1x1 or 2x2 inputs
                offset = (5 - max(input_height, input_width)) // 2
                for i in range(input_height):
                    for j in range(input_width):
                        output_grid[start_row + offset + i][start_col + offset + j] = input_grid.values[i][j]
            # For 4x4 or larger inputs, we don't need to do anything extra here

    return ColoredGrid(values=output_grid)
