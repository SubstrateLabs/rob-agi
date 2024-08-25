from rob_agi.colored_grid import ColoredGrid

def solve_5783df64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid of any size into a 3x3 grid containing the first 9 unique non-zero colors
    encountered in the input grid, preserving their order of appearance.

    The function scans the input grid from left to right, top to bottom, collecting unique non-zero colors.
    These colors are then used to fill a new 3x3 grid in the order they were found. If fewer than 9 unique
    colors are present, the remaining cells in the output grid are filled with 0 (black).

    Args:
    input_grid (ColoredGrid): The input grid of any size containing color values (0-9).

    Returns:
    ColoredGrid: A 3x3 grid containing the first 9 unique non-zero colors from the input grid.
    """
    color_order = []
    
    # Scan the input grid
    for row in input_grid.values:
        for color in row:
            if color != 0 and color not in color_order:
                color_order.append(color)
                if len(color_order) == 9:
                    break
        if len(color_order) == 9:
            break
    
    # Create and fill the output grid
    output_values = [[0 for _ in range(3)] for _ in range(3)]
    color_index = 0
    for i in range(3):
        for j in range(3):
            if color_index < len(color_order):
                output_values[i][j] = color_order[color_index]
                color_index += 1
    
    return ColoredGrid(values=output_values)
