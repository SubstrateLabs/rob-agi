from rob_agi.colored_grid import ColoredGrid

def solve_c87289bb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending sky blue (8) columns,
    creating zones around red (2) squares, and filling appropriate areas
    with sky blue, while preserving the original pattern above the red squares.

    1. Identify the "red row" containing red (2) cells.
    2. Preserve the original pattern above and including the red row.
    3. Identify vertical blue lines and zones between them.
    4. Process each zone: fill with sky blue if it contains red cells.
    5. Extend vertical blue lines to the bottom.
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

    # 3. Identify vertical blue lines and zones
    vertical_lines = [c for c in range(cols) if any(input_grid.values[r][c] == 8 for r in range(red_row_index + 1))]
    zones = [(vertical_lines[i], vertical_lines[i+1]) if i+1 < len(vertical_lines) else (vertical_lines[i], cols) 
             for i in range(len(vertical_lines))]
    zones = [(0, vertical_lines[0])] + zones if vertical_lines[0] != 0 else zones

    # 4. Process each zone
    for start, end in zones:
        if any(input_grid.values[red_row_index][c] == 2 for c in range(start, end)):
            for r in range(red_row_index + 1, rows):
                for c in range(start, end):
                    if new_grid.values[r][c] != 2:  # Don't overwrite red cells
                        new_grid.values[r][c] = 8

    # 5. Extend vertical blue lines
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
