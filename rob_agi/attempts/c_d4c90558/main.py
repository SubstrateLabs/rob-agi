from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d4c90558(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d4c90558 challenge by extracting the largest rectangular region for each color.
    
    The function scans the input grid, identifies the largest contiguous rectangular region
    for each unique color (excluding black and gray), reduces each region to a single row,
    and arranges these rows in the order of their first appearance in the input grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to process.
    
    Returns:
    ColoredGrid: A new grid where each row represents the largest region of a unique color.
    """
    def find_largest_region(start_r: int, start_c: int, color: int) -> Tuple[int, int, int]:
        max_width = 0
        max_height = 0
        for c in range(start_c, cols):
            if input_grid.get_cell(start_r, c) not in [color, 5]:
                break
            max_width += 1
        for r in range(start_r, rows):
            if all(input_grid.get_cell(r, c) in [color, 5] for c in range(start_c, start_c + max_width)):
                max_height += 1
            else:
                break
        return max_width, max_height, color

    rows, cols = input_grid.get_dimensions()
    processed_colors = set()
    extracted_regions = []

    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color not in [0, 5] and color not in processed_colors:
                width, height, color = find_largest_region(r, c, color)
                extracted_regions.append([color] * width)
                processed_colors.add(color)

    return ColoredGrid(values=extracted_regions)
