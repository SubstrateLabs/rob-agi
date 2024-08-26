from rob_agi.colored_grid import ColoredGrid

def solve_5783df64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid of any size into a 3x3 grid containing the first 9 unique non-zero colors
    encountered in the input grid, preserving their order of appearance.

    The function scans the input grid from top to bottom, left to right, collecting unique non-zero colors.
    These colors are then used to fill a new 3x3 grid in the order they were found, filling row by row.
    If fewer than 9 unique colors are present, the remaining cells in the output grid are filled with 0 (black).

    Args:
    input_grid (ColoredGrid): The input grid of any size containing color values (0-9).

    Returns:
    ColoredGrid: A 3x3 grid containing the first 9 unique non-zero colors from the input grid.
    """
    color_order = []
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    
    # Scan the input grid
    for r in range(rows):
        for c in range(cols):
            color = input_grid.values[r][c]
            if color != 0 and color not in color_order:
                color_order.append(color)
                if len(color_order) == 9:
                    break
        if len(color_order) == 9:
            break
    
    # Create and fill the output grid
    output_values = [[0 for _ in range(3)] for _ in range(3)]
    color_index = 0
    for r in range(3):
        for c in range(3):
            if color_index < len(color_order):
                output_values[r][c] = color_order[color_index]
                color_index += 1
    
    return ColoredGrid(values=output_values)
