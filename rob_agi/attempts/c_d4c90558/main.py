from rob_agi.colored_grid import ColoredGrid
from typing import Dict, List

def solve_d4c90558(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d4c90558 challenge by extracting the largest contiguous width for each color.
    
    The function scans the input grid row by row, identifies the largest contiguous width
    for each unique color (excluding black and gray), and arranges these widths in the order
    of their first appearance in the input grid. All rows are padded to have the same length
    as the overall largest width found.
    
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
    color_widths: Dict[int, int] = {}
    color_order: List[int] = []
    max_width = 0

    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color not in [0, 5]:
                if color not in color_widths:
                    color_order.append(color)
                    color_widths[color] = 0
                width = find_contiguous_width(r, c, color)
                color_widths[color] = max(color_widths[color], width)
                max_width = max(max_width, width)

    output_rows = []
    for color in color_order:
        width = color_widths[color]
        row = [color] * width + [0] * (max_width - width)
        output_rows.append(row)

    return ColoredGrid(values=output_rows)
