from rob_agi.colored_grid import ColoredGrid

def solve_3f23242b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a pattern around each green (3) cell.
    The pattern consists of a 5x5 gray (5) frame, surrounded by red (2) borders on three sides,
    and a sky blue (8) line extending to the grid edges. The pattern orientation
    (horizontal or vertical) is determined based on the green cell's position relative to the grid center.
    Red borders are extended until they reach a non-black cell or the grid edge.
    Colors are applied with the priority: Green > Sky blue > Red > Gray > Black.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    def draw_pattern(center_row, center_col, is_horizontal):
        # Draw sky blue line
        if is_horizontal:
            for col in range(width):
                if grid.values[center_row][col] != 3:
                    grid.values[center_row][col] = 8
        else:
            for row in range(height):
                if grid.values[row][center_col] != 3:
                    grid.values[row][center_col] = 8

        # Draw 5x5 gray frame
        for i in range(max(0, center_row - 2), min(height, center_row + 3)):
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                if grid.values[i][j] == 0:
                    grid.values[i][j] = 5

        # Draw initial red border
        if is_horizontal:
            for i in range(max(0, center_row - 2), min(height, center_row + 3)):
                if i != center_row:
                    if grid.values[i][max(0, center_col - 2)] == 0:
                        grid.values[i][max(0, center_col - 2)] = 2
                    if grid.values[i][min(width - 1, center_col + 2)] == 0:
                        grid.values[i][min(width - 1, center_col + 2)] = 2
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                if grid.values[min(height - 1, center_row + 2)][j] == 0:
                    grid.values[min(height - 1, center_row + 2)][j] = 2
        else:
            for j in range(max(0, center_col - 2), min(width, center_col + 3)):
                if j != center_col:
                    if grid.values[max(0, center_row - 2)][j] == 0:
                        grid.values[max(0, center_row - 2)][j] = 2
                    if grid.values[min(height - 1, center_row + 2)][j] == 0:
                        grid.values[min(height - 1, center_row + 2)][j] = 2
            for i in range(max(0, center_row - 2), min(height, center_row + 3)):
                if grid.values[i][min(width - 1, center_col + 2)] == 0:
                    grid.values[i][min(width - 1, center_col + 2)] = 2

    def extend_red_borders():
        def extend_border(r, c, dr, dc):
            while 0 <= r < height and 0 <= c < width:
                if grid.values[r][c] != 0:
                    if grid.values[r][c] == 2:
                        r += dr
                        c += dc
                    else:
                        break
                else:
                    grid.values[r][c] = 2
                    break

        for r in range(height):
            for c in range(width):
                if grid.values[r][c] == 2:
                    extend_border(r+1, c, 1, 0)  # Down
                    extend_border(r-1, c, -1, 0)  # Up
                    extend_border(r, c+1, 0, 1)  # Right
                    extend_border(r, c-1, 0, -1)  # Left

    # Find green cells and apply pattern
    for row in range(height):
        for col in range(width):
            if grid.values[row][col] == 3:
                # Determine orientation based on position relative to grid center
                is_horizontal = col >= width // 2
                draw_pattern(row, col, is_horizontal)

    # Extend red borders
    extend_red_borders()

    return grid
