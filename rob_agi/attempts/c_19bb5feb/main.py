from rob_agi.colored_grid import ColoredGrid

def solve_19bb5feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 19bb5feb challenge by transforming the input grid into a 2x2 output grid.
    
    The solution involves:
    1. Scanning the input grid to find 2x2 colored squares (excluding black and sky blue).
    2. Creating a 2x2 output grid where:
       - The top-left cell is the color with the lowest value.
       - The top-right cell is the color with the highest value.
       - The bottom-left cell is the middle color value (if 3 colors are found) or black (if 2 or fewer colors).
       - The bottom-right cell is the second highest color value (if at least 2 colors) or black (if only 1 color).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed 2x2 output grid.
    """
    found_colors = set()
    search_colors = set(range(1, 8))  # Colors 1 to 7, excluding 0 and 8

    rows, cols = input_grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            color = input_grid.values[r][c]
            if color in search_colors:
                if (color == input_grid.values[r][c+1] == 
                    input_grid.values[r+1][c] == input_grid.values[r+1][c+1]):
                    found_colors.add(color)

    output_values = [[0, 0], [0, 0]]
    if found_colors:
        sorted_colors = sorted(found_colors)
        output_values[0][0] = sorted_colors[0]  # Top-left: lowest color value
        output_values[0][1] = sorted_colors[-1]  # Top-right: highest color value
        if len(sorted_colors) >= 2:
            output_values[1][1] = sorted_colors[-2]  # Bottom-right: second highest color value
        if len(sorted_colors) == 3:
            output_values[1][0] = sorted_colors[1]  # Bottom-left: middle color value (only if 3 colors)

    return ColoredGrid(values=output_values)
