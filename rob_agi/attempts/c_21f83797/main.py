from rob_agi.colored_grid import ColoredGrid

def solve_21f83797(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red cross based on two red dots
    and filling a blue rectangle between them.

    1. Finds the two red dots in the input grid.
    2. Creates a red cross spanning the entire grid through these dots.
    3. Calculates and fills a blue rectangle between the cross arms.
    4. Returns the transformed grid.
    """
    # Initialize a new 13x13 grid filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])

    # Find the red dots
    red_dots = []
    for r in range(13):
        for c in range(13):
            if input_grid.values[r][c] == 2:
                red_dots.append((r, c))

    row1, col1 = red_dots[0]
    row2, col2 = red_dots[1]

    # Ensure row1 <= row2
    if row1 > row2:
        row1, row2 = row2, row1
        col1, col2 = col2, col1

    # Create the red cross
    for r in range(13):
        output_grid.values[row1][r] = 2
        output_grid.values[row2][r] = 2
    for c in range(13):
        output_grid.values[c][col1] = 2
        output_grid.values[c][col2] = 2

    # Calculate blue rectangle dimensions
    top = row1 + 1
    height = row2 - row1 - 1
    left = min(col1, col2) + 1
    right = max(col1, col2) - 1

    # Fill the blue rectangle
    for r in range(top, top + height):
        for c in range(left, right + 1):
            if output_grid.values[r][c] != 2:
                output_grid.values[r][c] = 1

    return output_grid
