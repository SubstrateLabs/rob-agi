from rob_agi.colored_grid import ColoredGrid

def solve_d4a91cb9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by drawing a path from the source (8) to the destination (2).
    The path is drawn first vertically from the source's y-coordinate to the destination's y-coordinate,
    then horizontally from the source's x-coordinate to the destination's x-coordinate.
    The path is drawn with color 4 (yellow), replacing 0s (black) along the way.
    """
    # Find source (8) and destination (2) coordinates
    source, destination = None, None
    for y, row in enumerate(input_grid.values):
        for x, cell in enumerate(row):
            if cell == 8:
                source = (x, y)
            elif cell == 2:
                destination = (x, y)
    
    if not source or not destination:
        return input_grid  # Return original grid if source or destination is missing

    # Create a copy of the input grid
    result = input_grid.deep_copy()

    # Draw vertical path
    start_y, end_y = min(source[1], destination[1]), max(source[1], destination[1])
    for y in range(start_y, end_y + 1):
        if result.get_cell(y, source[0]) == 0:
            result.set_cell(y, source[0], 4)

    # Draw horizontal path
    start_x, end_x = min(source[0], destination[0]), max(source[0], destination[0])
    for x in range(start_x, end_x + 1):
        if result.get_cell(destination[1], x) == 0:
            result.set_cell(destination[1], x, 4)

    return result
