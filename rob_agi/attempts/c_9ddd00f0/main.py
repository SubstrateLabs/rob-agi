from rob_agi.colored_grid import ColoredGrid

def solve_9ddd00f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by mirroring the pattern vertically.
    
    The function processes each column from right to left:
    1. Copies the entire column from input to output.
    2. Identifies the topmost colored (non-black) cell in the column.
    3. Mirrors the pattern from the bottom to the top of the column,
       up to the same distance as from the bottom to the topmost colored cell.
    
    This preserves the original pattern on the right and bottom sides,
    while creating a vertically mirrored pattern on the top side.
    """
    height, width = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    for col in range(width - 1, -1, -1):
        topmost_colored_row = None
        for row in range(height - 1, -1, -1):
            if input_grid.values[row][col] != 0 and topmost_colored_row is None:
                topmost_colored_row = row

        if topmost_colored_row is not None:
            distance = height - 1 - topmost_colored_row
            for row in range(distance + 1):
                output_grid.values[row][col] = input_grid.values[height - 1 - row][col]

    return output_grid
