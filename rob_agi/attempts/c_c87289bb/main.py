from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating enclosures around red (2) squares, and filling appropriate areas
    with sky blue, while preserving the original pattern above the red squares.

    1. Identify the row with red squares and existing sky blue columns.
    2. Extend original sky blue columns downward.
    3. Create enclosures for red sections.
    4. Handle overlapping enclosures.
    5. Fill in the enclosures.
    6. Connect the bottom row.
    7. Preserve the original pattern above the red row.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # 1. Identify the red row and existing sky blue columns
    red_row_index = next(i for i, row in enumerate(input_grid.values) if 2 in row)
    existing_blue_cols = [c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row_index))]

    # 2. Extend original sky blue columns downward
    for c in existing_blue_cols:
        for r in range(red_row_index, rows):
            new_grid.values[r][c] = 8

    # 3. Create enclosures for red sections
    red_sections = []
    start = None
    for c in range(cols):
        if input_grid.values[red_row_index][c] == 2:
            if start is None:
                start = c
        elif start is not None:
            red_sections.append((start, c - 1))
            start = None
    if start is not None:
        red_sections.append((start, cols - 1))

    # 4. Handle overlapping enclosures and fill them
    for start, end in red_sections:
        left = max([c for c in existing_blue_cols if c < start], default=-1)
        right = min([c for c in existing_blue_cols if c > end], default=cols)
        
        # Fill the enclosure
        for r in range(red_row_index, rows):
            for c in range(left + 1, right):
                if new_grid.values[r][c] != 2:  # Don't overwrite red cells
                    new_grid.values[r][c] = 8

    # 5. Connect the bottom row
    bottom_row = rows - 1
    left_edge = min(c for c in range(cols) if new_grid.values[bottom_row][c] == 8)
    right_edge = max(c for c in range(cols) if new_grid.values[bottom_row][c] == 8)
    for c in range(left_edge, right_edge + 1):
        new_grid.values[bottom_row][c] = 8

    # 6. Preserve original pattern above the red row
    for r in range(red_row_index):
        new_grid.values[r] = input_grid.values[r][:]

    return new_grid
