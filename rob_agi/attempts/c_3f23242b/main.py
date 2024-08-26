from rob_agi.colored_grid import ColoredGrid

def solve_3f23242b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a pattern around each green (3) cell.
    The pattern consists of a 5x5 gray (5) frame, surrounded by red (2) borders,
    and a sky blue (8) line extending to the grid edges. The pattern orientation
    (horizontal or vertical) is determined based on available space.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    def draw_pattern(center_row, center_col, is_horizontal):
        # Draw 5x5 gray frame
        for i in range(max(0, center_row - 2), min(height, center_row + 3)):
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                grid.values[i][j] = 5

        # Draw red frame
        for i in range(max(0, center_row - 3), min(height, center_row + 4)):
            for j in range(max(0, center_col - 3), min(width, center_col + 4)):
                if i == center_row - 3 or i == center_row + 3 or j == center_col - 3 or j == center_col + 3:
                    if 0 <= i < height and 0 <= j < width:
                        grid.values[i][j] = 2

        # Draw sky blue line
        if is_horizontal:
            row = min(height - 1, center_row + 3) if center_row + 3 < height else max(0, center_row - 3)
            for col in range(width):
                grid.values[row][col] = 8
        else:
            col = min(width - 1, center_col + 3) if center_col + 3 < width else max(0, center_col - 3)
            for row in range(height):
                grid.values[row][col] = 8

        # Ensure center remains green
        grid.values[center_row][center_col] = 3

    # Find green cells and apply pattern
    for row in range(height):
        for col in range(width):
            if grid.values[row][col] == 3:
                # Determine orientation based on available space
                space_horizontal = min(col, width - 1 - col)
                space_vertical = min(row, height - 1 - row)
                is_horizontal = space_horizontal >= space_vertical
                draw_pattern(row, col, is_horizontal)

    return grid
