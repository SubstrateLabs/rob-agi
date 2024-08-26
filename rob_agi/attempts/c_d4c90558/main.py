from rob_agi.colored_grid import ColoredGrid
from typing import Dict, List, Tuple

def solve_d4c90558(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d4c90558 challenge by extracting the largest square region for each color.
    
    The function scans the input grid from top-left to bottom-right, identifies the largest
    contiguous square region for each unique color (excluding black and gray), reduces each
    region to a single row, and arranges these rows in the order of their first appearance
    in the input grid. All rows are padded to have the same length as the overall largest
    square found.
    
    Args:
    input_grid (ColoredGrid): The input grid to process.
    
    Returns:
    ColoredGrid: A new grid where each row represents the largest square region of a unique color,
                 padded to ensure all rows have the same length as the largest square found.
    """
    def find_largest_square(row: int, col: int, color: int) -> int:
        size = 0
        while True:
            size += 1
            if row + size > rows or col + size > cols:
                return size - 1
            if any(input_grid.get_cell(r, c) not in [color, 5] 
                   for r in range(row, row + size) 
                   for c in range(col, col + size)):
                return size - 1

    rows, cols = input_grid.get_dimensions()
    color_sizes: Dict[int, int] = {}
    color_order: List[int] = []
    max_square_size = 0

    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color not in [0, 5]:
                if color not in color_sizes:
                    color_order.append(color)
                    color_sizes[color] = 0
                if r == 0 or c == 0 or input_grid.get_cell(r-1, c) != color or input_grid.get_cell(r, c-1) != color:
                    size = find_largest_square(r, c, color)
                    color_sizes[color] = max(color_sizes[color], size)
                    max_square_size = max(max_square_size, size)

    output_rows = []
    for color in color_order:
        size = color_sizes[color]
        row = [color] * size + [0] * (max_square_size - size)
        output_rows.append(row)

    return ColoredGrid(values=output_rows)
