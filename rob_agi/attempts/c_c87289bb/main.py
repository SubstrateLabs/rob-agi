from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating enclosures around red (2) squares, and filling appropriate areas
    with sky blue, while preserving the original pattern above the red squares.

    1. Identify the "red row" containing red (2) cells.
    2. Preserve the original pattern above and including the red row.
    3. Identify vertical blue lines.
    4. Create enclosures around red segments.
    5. Extend original blue lines to the bottom.
    6. Connect the bottom row with sky blue.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # 1. Identify the "red row"
    red_row_index = next(i for i, row in enumerate(input_grid.values) if 2 in row)

    # 2. Preserve the original pattern
    for r in range(red_row_index + 1):
        new_grid.values[r] = input_grid.values[r][:]

    # 3. Identify vertical blue lines
    vertical_lines = [c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row_index + 1))]

    # 4. Create enclosures around red segments
    red_segments = []
    start = None
    for c in range(cols):
        if input_grid.values[red_row_index][c] == 2:
            if start is None:
                start = c
        elif start is not None:
            red_segments.append((start, c - 1))
            start = None
    if start is not None:
        red_segments.append((start, cols - 1))

    for start, end in red_segments:
        left_boundary = max([c for c in vertical_lines if c < start] + [0])
        right_boundary = min([c for c in vertical_lines if c > end] + [cols - 1])
        for r in range(red_row_index + 1, rows):
            for c in range(left_boundary, right_boundary + 1):
                new_grid.values[r][c] = 8

    # 5. Extend original blue lines
    for c in vertical_lines:
        for r in range(red_row_index + 1, rows):
            new_grid.values[r][c] = 8

    # 6. Connect the bottom row
    bottom_row = rows - 1
    left_edge = min(c for c in range(cols) if new_grid.values[bottom_row][c] == 8)
    right_edge = max(c for c in range(cols) if new_grid.values[bottom_row][c] == 8)
    for c in range(left_edge, right_edge + 1):
        new_grid.values[bottom_row][c] = 8

    return new_grid
