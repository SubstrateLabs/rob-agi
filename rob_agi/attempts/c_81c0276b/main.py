from collections import OrderedDict
from rob_agi.colored_grid import ColoredGrid

def solve_81c0276b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting 2x2 colored squares and arranging them in a compact output grid.

    The function works as follows:
    1. Identifies the frame color in the input grid.
    2. Scans the grid in a snake-like pattern from top-left to bottom-right to find 2x2 colored squares.
    3. Records the frequency of each color in the order of first appearance.
    4. Creates an output grid where each row corresponds to a unique color, ordered by frequency (descending) and then by first appearance.
    5. Fills the output grid with the colors found, maintaining their order and frequency.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    frame_color = next(color for color in input_grid.values[0] if color != 0)
    
    color_info = OrderedDict()
    
    rows, cols = input_grid.get_dimensions()
    for i in range(rows):
        row_range = range(cols) if i % 2 == 0 else range(cols - 1, -1, -1)
        for j in row_range:
            color = input_grid.values[i][j]
            if color != 0 and color != frame_color:
                if (i + 1 < rows and j + 1 < cols and
                    all(input_grid.values[i+di][j+dj] == color 
                        for di in range(2) for dj in range(2))):
                    if color not in color_info:
                        color_info[color] = 1
                    else:
                        color_info[color] += 1
    
    # Sort colors by frequency (descending) and then by order of appearance
    sorted_colors = sorted(color_info.items(), key=lambda x: (-x[1], list(color_info.keys()).index(x[0])))
    
    output_rows = len(sorted_colors)
    output_cols = max(color_info.values())
    
    output_values = [[0] * output_cols for _ in range(output_rows)]
    for i, (color, freq) in enumerate(sorted_colors):
        output_values[i][:freq] = [color] * freq
    
    return ColoredGrid(values=output_values)
