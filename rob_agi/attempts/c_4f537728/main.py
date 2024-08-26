from rob_agi.colored_grid import ColoredGrid

def solve_4f537728(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating seed colors (red and green) according to specific rules:
    - Red (2) propagates both horizontally and vertically, replacing non-black cells.
    - Green (3) propagates only vertically, replacing non-black cells.
    - Other colors remain unchanged unless replaced by propagating colors.
    - Black (0) cells are never changed and act as barriers to propagation.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the color propagation rules.
    """
    def propagate_color(grid, row, col, color):
        rows, cols = grid.get_dimensions()
        # Vertical propagation (for both red and green)
        for r in range(rows):
            if grid.values[r][col] != 0:  # Skip black cells
                grid.values[r][col] = color
        # Horizontal propagation (only for red)
        if color == 2:
            for c in range(cols):
                if grid.values[row][c] != 0:  # Skip black cells
                    grid.values[row][c] = color

    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    for row in range(rows):
        for col in range(cols):
            color = new_grid.values[row][col]
            if color in [2, 3]:  # Red or Green
                propagate_color(new_grid, row, col, color)

    return new_grid
