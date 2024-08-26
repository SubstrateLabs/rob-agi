from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating new sky blue columns adjacent to red (2) squares, and filling
    appropriate areas with sky blue, while preserving the original pattern
    above the red squares.

    1. Identify the row with red squares and existing sky blue columns.
    2. Create new sky blue columns adjacent to red squares.
    3. Extend all sky blue columns downward.
    4. Fill areas below red squares with sky blue.
    5. Preserve the original pattern above the red row.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # 1. Identify the red row and existing sky blue columns
    red_row_index = next(i for i, row in enumerate(input_grid.values) if 2 in row)
    existing_blue_cols = set(c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row_index)))

    # 2. Process red squares and create new sky blue columns
    new_blue_cols = set()
    red_squares = []
    for c in range(cols):
        if input_grid.values[red_row_index][c] == 2:
            red_squares.append(c)
            if c > 0 and c - 1 not in existing_blue_cols:
                new_blue_cols.add(c - 1)
            if c < cols - 1 and c + 1 not in existing_blue_cols:
                new_blue_cols.add(c + 1)

    # 3. Combine existing and new sky blue columns
    all_blue_cols = existing_blue_cols.union(new_blue_cols)

    # 4. Process the red row and below
    for r in range(red_row_index, rows):
        for c in range(cols):
            if c in all_blue_cols or (input_grid.values[r][c] == 8 and c in existing_blue_cols):
                new_grid.values[r][c] = 8
            else:
                new_grid.values[r][c] = input_grid.values[r][c]

    # 5. Extend existing sky blue columns
    for c in existing_blue_cols:
        for r in range(red_row_index):
            new_grid.values[r][c] = 8

    # 6. Fill areas below red squares
    for red_col in red_squares:
        left_blue = max((c for c in all_blue_cols if c < red_col), default=-1)
        right_blue = min((c for c in all_blue_cols if c > red_col), default=cols)
        for r in range(red_row_index + 1, rows):
            for c in range(left_blue + 1, right_blue):
                new_grid.values[r][c] = 8

    return new_grid
