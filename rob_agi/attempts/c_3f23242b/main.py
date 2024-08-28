from rob_agi.colored_grid import ColoredGrid

def solve_3f23242b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a pattern around each green (3) cell.
    The pattern consists of a 5x5 gray (5) frame, surrounded by red (2) borders on three sides,
    and a sky blue (8) line extending to the grid edges. The pattern orientation
    (horizontal or vertical) is determined based on the green cell's position relative to the grid center.
    Red borders are extended to the grid edge in the direction away from the sky blue line.
    Colors are applied with the priority: Green > Sky blue > Red > Gray > Black.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()
    mid_row, mid_col = height // 2, width // 2

    def color_priority(color):
        return {3: 4, 8: 3, 2: 2, 5: 1, 0: 0}.get(color, 0)

    def set_color(row, col, color):
        if 0 <= row < height and 0 <= col < width:
            if color_priority(color) > color_priority(grid.values[row][col]):
                grid.values[row][col] = color

    def get_pattern_orientation(row, col):
        return 'vertical' if abs(row - mid_row) > abs(col - mid_col) else 'horizontal'

    def draw_sky_blue_line(row, col, orientation):
        if orientation == 'horizontal':
            for c in range(width):
                set_color(row, c, 8)
        else:
            for r in range(height):
                set_color(r, col, 8)

    def create_gray_frame(row, col):
        for i in range(max(0, row - 2), min(height, row + 3)):
            for j in range(max(0, col - 2), min(width, col + 3)):
                set_color(i, j, 5)

    def add_red_borders(row, col, orientation):
        if orientation == 'horizontal':
            for i in range(max(0, row - 2), min(height, row + 3)):
                set_color(i, max(0, col - 2), 2)
                set_color(i, min(width - 1, col + 2), 2)
            if row >= mid_row:
                for j in range(max(0, col - 2), min(width, col + 3)):
                    set_color(max(0, row - 2), j, 2)
                for i in range(row - 2, -1, -1):
                    set_color(i, max(0, col - 2), 2)
                    set_color(i, min(width - 1, col + 2), 2)
            else:
                for j in range(max(0, col - 2), min(width, col + 3)):
                    set_color(min(height - 1, row + 2), j, 2)
                for i in range(row + 3, height):
                    set_color(i, max(0, col - 2), 2)
                    set_color(i, min(width - 1, col + 2), 2)
        else:
            for j in range(max(0, col - 2), min(width, col + 3)):
                set_color(max(0, row - 2), j, 2)
                set_color(min(height - 1, row + 2), j, 2)
            if col >= mid_col:
                for i in range(max(0, row - 2), min(height, row + 3)):
                    set_color(i, max(0, col - 2), 2)
                for j in range(col - 2, -1, -1):
                    set_color(max(0, row - 2), j, 2)
                    set_color(min(height - 1, row + 2), j, 2)
            else:
                for i in range(max(0, row - 2), min(height, row + 3)):
                    set_color(i, min(width - 1, col + 2), 2)
                for j in range(col + 3, width):
                    set_color(max(0, row - 2), j, 2)
                    set_color(min(height - 1, row + 2), j, 2)

    # Find green cells and apply pattern
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] == 3:
                orientation = get_pattern_orientation(row, col)
                draw_sky_blue_line(row, col, orientation)
                create_gray_frame(row, col)
                add_red_borders(row, col, orientation)

    # Restore green cells
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] == 3:
                grid.values[row][col] = 3

    return grid
