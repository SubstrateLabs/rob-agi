from rob_agi.colored_grid import ColoredGrid

def solve_3391f8c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a vertical flip, swapping specific colors,
    and adjusting patterns to maintain their general shape and relative positions.
    The color swap pairs are:
    1 (blue) <-> 7 (orange), 1 (blue) <-> 8 (sky), 2 (red) <-> 3 (green).
    The swap for color 1 depends on the presence of 7 or 8 in the grid.
    If both 7 and 8 are present, 1 swaps with 7.
    Black (0) and other colors remain unchanged.
    Patterns are adjusted to fit available space while maintaining their general shape.
    """
    height, width = input_grid.get_dimensions()
    unique_colors = set(color for row in input_grid.values for color in row)

    # Determine color mapping
    color_map = {}
    if 1 in unique_colors:
        if 7 in unique_colors:
            color_map[1] = 7
            color_map[7] = 1
        elif 8 in unique_colors:
            color_map[1] = 8
            color_map[8] = 1
    if 2 in unique_colors and 3 in unique_colors:
        color_map[2] = 3
        color_map[3] = 2

    # Step 1: Vertical flip and color swap
    flipped_values = []
    for i in range(height - 1, -1, -1):
        row = [color_map.get(input_grid.get_cell(i, j), input_grid.get_cell(i, j)) for j in range(width)]
        flipped_values.append(row)

    # Step 2: Identify and adjust patterns
    output_grid = ColoredGrid(values=flipped_values)
    for color in unique_colors:
        if color == 0:
            continue
        regions = output_grid.find_connected_regions(color)
        for region in regions:
            adjust_pattern(output_grid, region)

    return output_grid

def adjust_pattern(grid: ColoredGrid, region: List[Tuple[int, int]]):
    """Adjusts a pattern to fit available space while maintaining its general shape."""
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Check if pattern can be expanded vertically
    if min_row > 0 and all(grid.get_cell(min_row - 1, c) == 0 for c in range(min_col, max_col + 1)):
        for r, c in sorted(region):
            grid.set_cell(r - 1, c, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
    elif max_row < grid.num_rows - 1 and all(grid.get_cell(max_row + 1, c) == 0 for c in range(min_col, max_col + 1)):
        for r, c in sorted(region, reverse=True):
            grid.set_cell(r + 1, c, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)

    # Check if pattern can be expanded horizontally
    if min_col > 0 and all(grid.get_cell(r, min_col - 1) == 0 for r in range(min_row, max_row + 1)):
        for r, c in sorted(region, key=lambda x: x[1]):
            grid.set_cell(r, c - 1, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
    elif max_col < grid.num_cols - 1 and all(grid.get_cell(r, max_col + 1) == 0 for r in range(min_row, max_row + 1)):
        for r, c in sorted(region, key=lambda x: x[1], reverse=True):
            grid.set_cell(r, c + 1, grid.get_cell(r, c))
            grid.set_cell(r, c, 0)
