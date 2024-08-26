from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating new sky blue columns to enclose red (2) squares, and filling
    appropriate areas with sky blue, while preserving the original pattern
    above the red squares and outside the fillable area.

    1. Identify the row with red squares and sky blue columns in the top section.
    2. Extend existing sky blue columns downward.
    3. Create new sky blue columns to enclose red squares.
    4. Fill areas between sky blue columns within the fillable area.
    5. Preserve the original pattern in the top section and outside the fillable area.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # Find the row with red squares
    red_row = next(i for i, row in enumerate(input_grid.values) if 2 in row)

    # Identify sky blue columns in the top section
    sky_blue_cols = [c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row))]

    # Define the fillable area
    fillable_left = min(sky_blue_cols) if sky_blue_cols else 0
    fillable_right = max(sky_blue_cols) if sky_blue_cols else cols - 1

    # Process the red row and create new sky blue columns
    red_groups = []
    start = None
    for c in range(cols):
        if input_grid.values[red_row][c] == 2:
            if start is None:
                start = c
        elif start is not None:
            red_groups.append((start, c - 1))
            start = None
    if start is not None:
        red_groups.append((start, cols - 1))

    for start, end in red_groups:
        left_col = max([c for c in sky_blue_cols if c < start] + [fillable_left - 1]) + 1
        right_col = min([c for c in sky_blue_cols if c > end] + [fillable_right + 1]) - 1
        sky_blue_cols.extend([left_col, right_col])

    # Sort and deduplicate sky blue columns
    sky_blue_cols = sorted(set(sky_blue_cols))

    # Process the bottom section
    for r in range(red_row, rows):
        for c in range(cols):
            if c in sky_blue_cols:
                new_grid.values[r][c] = 8
            elif fillable_left <= c <= fillable_right:
                left_blue = max([col for col in sky_blue_cols if col < c] + [fillable_left - 1])
                right_blue = min([col for col in sky_blue_cols if col > c] + [fillable_right + 1])
                if left_blue >= fillable_left and right_blue <= fillable_right:
                    new_grid.values[r][c] = 8 if input_grid.values[red_row][c] == 8 else input_grid.values[r][c]

    return new_grid
