from rob_agi.colored_grid import ColoredGrid

def solve_3391f8c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 180-degree rotation and swapping specific colors.
    The color swap pairs are:
    1 (blue) <-> 7 (orange), 1 (blue) <-> 8 (sky), 2 (red) <-> 3 (green).
    The swap for color 1 depends on the presence of 7 or 8 in the grid.
    If both 7 and 8 are present, 1 swaps with 7.
    Black (0) and other colors remain unchanged.
    The transformation is applied globally based on the colors present in the entire grid.
    """
    height, width = input_grid.get_dimensions()
    unique_colors = set(color for row in input_grid.values for color in row)

    # Determine color mapping
    color_map = {}
    if 1 in unique_colors:
        if 7 in unique_colors:
            color_map[1] = 7
            color_map[7] = 1
        elif 8 in unique_colors:
            color_map[1] = 8
            color_map[8] = 1
    if 2 in unique_colors and 3 in unique_colors:
        color_map[2] = 3
        color_map[3] = 2

    new_values = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            new_i, new_j = height - 1 - i, width - 1 - j
            original_color = input_grid.get_cell(i, j)
            new_color = color_map.get(original_color, original_color)
            new_values[new_i][new_j] = new_color

    return ColoredGrid(values=new_values)
