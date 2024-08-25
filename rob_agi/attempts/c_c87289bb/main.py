from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns
    and filling areas between them with sky blue, while preserving the original
    pattern above the red (2) squares and the red squares themselves.

    1. Identify the row containing red squares.
    2. Identify sky blue columns from the top part of the grid.
    3. Extend sky blue columns downwards.
    4. Fill areas between sky blue columns with sky blue, except for red squares.
    5. Preserve columns that don't contain sky blue in the top part.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # Find the row with red squares
    red_row = next(i for i, row in enumerate(input_grid.values) if 2 in row)

    # Identify sky blue columns
    sky_blue_cols = [c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row + 1))]

    # Extend sky blue columns and fill areas between them
    for r in range(red_row + 1, rows):
        for c in range(cols):
            if c in sky_blue_cols:
                new_grid.values[r][c] = 8
            elif sky_blue_cols and sky_blue_cols[0] < c < sky_blue_cols[-1]:
                new_grid.values[r][c] = 8 if input_grid.values[r][c] != 2 else 2
            else:
                new_grid.values[r][c] = input_grid.values[r][c]

    return new_grid
