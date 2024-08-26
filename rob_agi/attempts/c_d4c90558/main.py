from rob_agi.colored_grid import ColoredGrid
from typing import Dict, List, Tuple

def solve_d4c90558(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d4c90558 challenge by extracting the largest contiguous width for each color.
    
    The function scans the input grid row by row, identifies the largest contiguous width
    for each unique color (excluding black and gray), and arranges these widths in the order
    of their first appearance in the input grid. The output grid is created with each row
    representing a color, and all rows are padded to have the same length as the overall
    largest width found. The colors are arranged in order of their topmost occurrence,
    with ties broken by leftmost position. Gray (5) is treated as a continuation of the
    current color when calculating contiguous widths.
    
    Args:
    input_grid (ColoredGrid): The input grid to process.
    
    Returns:
    ColoredGrid: A new grid where each row represents the largest contiguous width of a unique color,
                 padded to ensure all rows have the same length as the largest width found.
    """
    def find_contiguous_width(row: int, col: int, color: int) -> int:
        width = 0
        while col < cols and input_grid.get_cell(row, col) in [color, 5]:
            width += 1
            col += 1
        return width

    rows, cols = input_grid.get_dimensions()
    color_info: Dict[int, Tuple[int, int, int]] = {}  # color: (top, left, width)
    color_order: List[int] = []
    max_width = 0

    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color not in [0, 5]:
                if color not in color_info:
                    color_info[color] = (r, c, 0)
                    color_order.append(color)
                top, left, current_width = color_info[color]
                width = find_contiguous_width(r, c, color)
                if width > current_width:
                    color_info[color] = (top, left, width)
                max_width = max(max_width, width)

    # Sort colors based on their topmost occurrence, then leftmost
    color_order.sort(key=lambda x: (color_info[x][0], color_info[x][1]))

    output_rows = []
    for color in color_order:
        _, _, width = color_info[color]
        row = [color] * width + [0] * (max_width - width)
        output_rows.append(row)

    return ColoredGrid(values=output_rows)
