from rob_agi.colored_grid import ColoredGrid

def solve_3391f8c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a double flip (horizontal and vertical)
    and swapping specific colors. The color swap pairs are:
    1 (blue) <-> 7 (orange), 2 (red) <-> 3 (green), 8 (sky) <-> 1 (blue).
    Black (0) and other colors remain unchanged.
    """
    color_swap = {1: 7, 7: 1, 2: 3, 3: 2, 8: 1}

    def swap_color(color):
        return color_swap.get(color, color)

    height, width = input_grid.get_dimensions()
    new_values = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            new_i, new_j = height - 1 - i, width - 1 - j
            original_color = input_grid.get_cell(i, j)
            new_color = swap_color(original_color)
            new_values[new_i][new_j] = new_color

    return ColoredGrid(values=new_values)
