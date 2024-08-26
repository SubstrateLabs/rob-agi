from rob_agi.colored_grid import ColoredGrid

def solve_4f537728(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating seed colors (red and green) according to specific rules:
    - Red (2) propagates both horizontally and vertically, replacing non-black cells.
    - Green (3) propagates vertically and horizontally only in its original rows, replacing non-black cells.
    - Other colors remain unchanged unless replaced by propagating colors.
    - Black (0) cells are never changed and act as barriers to propagation.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the color propagation rules.
    """
    def propagate_horizontally(grid, row, color):
        for c in range(len(grid.values[row])):
            if grid.values[row][c] != 0:  # Skip black cells
                grid.values[row][c] = color

    def propagate_vertically(grid, col, color):
        for r in range(len(grid.values)):
            if grid.values[r][col] != 0:  # Skip black cells
                grid.values[r][col] = color

    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    red_cells = []
    green_cells = []

    # Scan for red and green cells
    for row in range(rows):
        for col in range(cols):
            if new_grid.values[row][col] == 2:  # Red
                red_cells.append((row, col))
            elif new_grid.values[row][col] == 3:  # Green
                green_cells.append((row, col))

    # Process red cells
    for row, col in red_cells:
        propagate_horizontally(new_grid, row, 2)
        propagate_vertically(new_grid, col, 2)

    # Process green cells
    for row, col in green_cells:
        propagate_horizontally(new_grid, row, 3)
        propagate_vertically(new_grid, col, 3)

    return new_grid
