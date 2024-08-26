from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points in quadrants.
    
    The function divides the grid into four quadrants, identifies seed points (non-gray, non-black cells)
    in each quadrant, and expands colors within their respective quadrants. The transformation respects
    the precedence of colors based on their quadrant position (top-left, top-right, bottom-left, bottom-right)
    and preserves the original structure including black cells and borders.
    
    Steps:
    1. Divide the grid into four quadrants
    2. Find seed points in each quadrant
    3. Expand colors within quadrants, respecting precedence
    4. Handle quadrants without seed points
    5. Preserve original black cells and borders
    6. Maintain remaining gray areas not claimed by any expansion
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2
    
    quadrants = [
        (0, 0, center_row, center_col),  # top-left
        (0, center_col, center_row, cols),  # top-right
        (center_row, 0, rows, center_col),  # bottom-left
        (center_row, center_col, rows, cols)  # bottom-right
    ]
    
    for top, left, bottom, right in quadrants:
        seed_point = find_seed_point(output_grid, top, left, bottom, right)
        if seed_point:
            color, _, _ = seed_point
            fill_quadrant(output_grid, color, top, left, bottom, right)
    
    preserve_original_structure(output_grid, input_grid)
    return output_grid

def find_seed_point(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> Tuple[int, int, int]:
    for r in range(top, bottom):
        for c in range(left, right):
            color = grid.get_cell(r, c)
            if color not in [0, 5]:  # Not black or gray
                return (color, r, c)
    return None

def fill_quadrant(grid: ColoredGrid, color: int, top: int, left: int, bottom: int, right: int):
    for r in range(top, bottom):
        for c in range(left, right):
            if grid.get_cell(r, c) in [0, 5]:  # Only fill black or gray cells
                grid.set_cell(r, c, color)

def preserve_original_structure(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = output_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0:  # If originally black
                output_grid.set_cell(r, c, 0)  # Keep it black
    # Ensure border is black
    for r in range(rows):
        output_grid.set_cell(r, 0, 0)
        output_grid.set_cell(r, cols-1, 0)
    for c in range(cols):
        output_grid.set_cell(0, c, 0)
        output_grid.set_cell(rows-1, c, 0)
