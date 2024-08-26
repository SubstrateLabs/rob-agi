from rob_agi.colored_grid import ColoredGrid

def solve_3f23242b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a pattern around each green (3) cell.
    The pattern consists of a 5x5 gray (5) frame, surrounded by red (2) borders on three sides,
    and a sky blue (8) line extending to the grid edges. The pattern orientation
    (horizontal or vertical) is determined based on the green cell's position relative to the grid center.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    def draw_pattern(center_row, center_col, is_horizontal):
        # Draw sky blue line
        if is_horizontal:
            for col in range(width):
                grid.values[center_row][col] = 8
        else:
            for row in range(height):
                grid.values[row][center_col] = 8

        # Draw 5x5 gray frame
        for i in range(max(0, center_row - 2), min(height, center_row + 3)):
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                grid.values[i][j] = 5

        # Draw red border
        if is_horizontal:
            for i in range(max(0, center_row - 2), min(height, center_row + 3)):
                if i != center_row:
                    grid.values[i][max(0, center_col - 3)] = 2
                    grid.values[i][min(width - 1, center_col + 3)] = 2
            for j in range(max(0, center_col - 3), min(width, center_col + 4)):
                grid.values[min(height - 1, center_row + 3)][j] = 2
        else:
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                if j != center_col:
                    grid.values[max(0, center_row - 3)][j] = 2
                    grid.values[min(height - 1, center_row + 3)][j] = 2
            for i in range(max(0, center_row - 3), min(height, center_row + 4)):
                grid.values[i][min(width - 1, center_col + 3)] = 2

        # Ensure center remains green
        grid.values[center_row][center_col] = 3

    # Find green cells and apply pattern
    for row in range(height):
        for col in range(width):
            if grid.values[row][col] == 3:
                # Determine orientation based on position relative to grid center
                is_horizontal = col > width // 2
                draw_pattern(row, col, is_horizontal)

    return grid
