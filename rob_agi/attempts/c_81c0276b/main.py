from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_81c0276b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting 2x2 colored squares and arranging them in a compact output grid.

    The function works as follows:
    1. Identifies the frame color in the input grid.
    2. Scans the grid from right to left, top to bottom, to find 2x2 colored squares.
    3. Records the frequency and first position of each color.
    4. Creates an output grid where each row corresponds to a unique color, sorted by appearance.
    5. Fills the output grid with the colors found, maintaining their order and frequency.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    frame_color = next(color for color in input_grid.values[0] if color != 0)
    
    color_info: Dict[int, List[int]] = {}  # {color: [frequency, first_row, first_col]}
    color_order: List[int] = []
    
    rows, cols = input_grid.get_dimensions()
    for col in range(cols - 1, -1, -1):
        for row in range(rows - 1):
            color = input_grid.values[row][col]
            if color != 0 and color != frame_color:
                if all(input_grid.values[row+i][col+j] == color for i in range(2) for j in range(2)):
                    if color not in color_info:
                        color_info[color] = [0, row, col]
                        color_order.append(color)
                    color_info[color][0] += 1
    
    sorted_colors = sorted(color_order, key=lambda c: (color_info[c][1], -color_info[c][2]))
    
    output_rows = len(sorted_colors)
    output_cols = max(color_info[color][0] for color in color_info) if color_info else 0
    
    output_values = [[0] * output_cols for _ in range(output_rows)]
    for i, color in enumerate(sorted_colors):
        freq = color_info[color][0]
        output_values[i][:freq] = [color] * freq
    
    return ColoredGrid(values=output_values)
