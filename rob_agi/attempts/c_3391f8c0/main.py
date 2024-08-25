from rob_agi.colored_grid import ColoredGrid

def solve_3391f8c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 180-degree rotation and swapping specific colors.
    The color swap pairs are:
    1 (blue) <-> 7 (orange), 2 (red) <-> 3 (green), 1 (blue) <-> 8 (sky).
    The swap for color 1 depends on the presence of 7 or 8 in the grid.
    Black (0) and other colors remain unchanged.
    """
    color_swap = {1: {7: 7, 8: 8}, 2: 3, 3: 2, 7: 1, 8: 1}

    def swap_color(color, original_pos):
        if color == 1:
            partner = input_grid.get_cell(height - 1 - original_pos[0], width - 1 - original_pos[1])
            if partner in [7, 8]:
                return partner
            elif 7 in input_grid.values:
                return 7
            elif 8 in input_grid.values:
                return 8
            else:
                return 1
        return color_swap.get(color, color)

    height, width = input_grid.get_dimensions()
    new_values = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            new_i, new_j = height - 1 - i, width - 1 - j
            original_color = input_grid.get_cell(i, j)
            new_color = swap_color(original_color, (i, j))
            new_values[new_i][new_j] = new_color

    return ColoredGrid(values=new_values)
