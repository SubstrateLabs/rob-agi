from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating enclosures around red (2) squares, and filling appropriate areas
    with sky blue, while preserving the original pattern above the red squares.

    1. Identify the row with red squares and existing sky blue columns.
    2. Create horizontal extension ranges based on red squares and blue columns.
    3. Merge overlapping ranges.
    4. Extend existing sky blue columns downward.
    5. Apply horizontal extensions to create enclosures.
    6. Preserve the original pattern above the red row.
    7. Ensure bottom row continuity of blue squares.

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

    # 2. Create horizontal extension ranges
    red_ranges = []
    start = None
    for c in range(cols):
        if input_grid.values[red_row_index][c] == 2:
            if start is None:
                start = c
        elif start is not None:
            red_ranges.append((start, c - 1))
            start = None
    if start is not None:
        red_ranges.append((start, cols - 1))

    # 3. Merge overlapping ranges and extend to blue columns or edges
    merged_ranges = []
    for start, end in sorted(red_ranges):
        if merged_ranges and start <= merged_ranges[-1][1] + 1:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    extension_ranges = []
    for start, end in merged_ranges:
        left = max((c for c in existing_blue_cols if c < start), default=-1) + 1
        right = min((c for c in existing_blue_cols if c > end), default=cols)
        extension_ranges.append((left, right))

    # 4. Extend existing blue columns
    for c in existing_blue_cols:
        for r in range(red_row_index, rows):
            new_grid.values[r][c] = 8

    # 5. Apply horizontal extensions
    for left, right in extension_ranges:
        for c in range(left, right):
            if new_grid.values[red_row_index][c] != 2:
                new_grid.values[red_row_index][c] = 8
        for r in range(red_row_index + 1, rows):
            for c in range(left, right):
                new_grid.values[r][c] = 8

    # 6. Preserve original pattern above red row
    for r in range(red_row_index):
        new_grid.values[r] = input_grid.values[r][:]

    # 7. Ensure bottom row continuity
    last_row = rows - 1
    for c in range(cols - 1, -1, -1):
        if new_grid.values[last_row][c] == 8:
            for cc in range(c - 1, -1, -1):
                if new_grid.values[last_row][cc] != 8:
                    break
                new_grid.values[last_row][cc] = 8

    return new_grid
