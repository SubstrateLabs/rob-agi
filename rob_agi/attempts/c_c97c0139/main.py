from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def find_red_lines(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    lines = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:  # Red cell
                if not lines or (r, c) != (lines[-1][2], lines[-1][3] + 1):  # New line
                    lines.append((r, c, r, c))
                else:  # Extend existing line
                    lines[-1] = (lines[-1][0], lines[-1][1], r, c)
    return lines

def is_within_diamond(row: int, col: int, line: Tuple[int, int, int, int], extension: int) -> bool:
    start_r, start_c, end_r, end_c = line
    if start_r == end_r:  # Horizontal line
        dist_to_line = abs(row - start_r)
        dist_to_end = min(abs(col - start_c), abs(col - end_c))
    else:  # Vertical line
        dist_to_line = abs(col - start_c)
        dist_to_end = min(abs(row - start_r), abs(row - end_r))
    
    return (dist_to_end + dist_to_line <= extension) or (dist_to_line < extension and dist_to_end < extension)

def solve_c97c0139(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c97c0139 challenge by creating rounded diamond-shaped sky blue (8) fields around red (2) lines.
    
    The solution follows these steps:
    1. Identify red lines in the input grid.
    2. For each red line, create a rounded diamond-shaped field around it.
    3. Set non-red cells within the diamond to sky blue.
    
    This approach works for both horizontal and vertical red lines of varying lengths.
    The diamond shape extends outwards from the red line, with its size determined by
    half the length of the red line. The shape is rounded at the corners to create a
    more natural-looking pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    red_lines = find_red_lines(input_grid)

    for line in red_lines:
        start_r, start_c, end_r, end_c = line
        line_length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
        extension = line_length // 2

        min_row = max(0, min(start_r, end_r) - extension)
        max_row = min(rows, max(start_r, end_r) + extension + 1)
        min_col = max(0, min(start_c, end_c) - extension)
        max_col = min(cols, max(start_c, end_c) + extension + 1)

        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                if output_grid.get_cell(r, c) != 2 and is_within_diamond(r, c, line, extension):
                    output_grid.set_cell(r, c, 8)

    return output_grid
