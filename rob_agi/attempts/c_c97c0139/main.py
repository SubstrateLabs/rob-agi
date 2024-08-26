from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def find_red_lines(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    lines = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:  # Red cell
                if not lines or (r, c) not in [(lines[-1][2], lines[-1][3] + 1), (lines[-1][2] + 1, lines[-1][3])]:  # New line
                    lines.append((r, c, r, c))
                else:  # Extend existing line
                    lines[-1] = (lines[-1][0], lines[-1][1], r, c)
    return lines

def calculate_extension(line_length: int) -> int:
    return max(1, min(line_length // 2, (line_length - 1) // 2))

def is_within_diamond(row: int, col: int, line: Tuple[int, int, int, int], extension: int) -> bool:
    start_r, start_c, end_r, end_c = line
    line_center_r, line_center_c = (start_r + end_r) / 2, (start_c + end_c) / 2
    
    if start_r == end_r:  # Horizontal line
        dist_to_line = abs(row - start_r)
        dist_to_center = abs(col - line_center_c)
    else:  # Vertical line
        dist_to_line = abs(col - start_c)
        dist_to_center = abs(row - line_center_r)
    
    line_length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
    max_dist = extension * (1 - dist_to_center / (line_length / 2 + extension))
    
    return dist_to_line <= max_dist

def solve_c97c0139(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c97c0139 challenge by creating diamond-shaped sky blue (8) fields around red (2) lines.
    
    The solution follows these steps:
    1. Identify red lines in the input grid (both horizontal and vertical).
    2. For each red line, create a diamond-shaped field around it.
    3. Set non-red cells within the diamond to sky blue.
    
    This approach works for both horizontal and vertical red lines of varying lengths.
    The diamond shape extends outwards from the red line, with its size determined by
    the length of the red line. The shape tapers at the ends to create a rounded diamond pattern.
    Very short lines (1-2 cells) still have a minimal diamond shape.
    The solution handles multiple non-intersecting lines correctly and allows diamonds to overlap.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    red_lines = find_red_lines(input_grid)

    for line in red_lines:
        start_r, start_c, end_r, end_c = line
        line_length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
        extension = calculate_extension(line_length)

        min_row = max(0, min(start_r, end_r) - extension)
        max_row = min(rows - 1, max(start_r, end_r) + extension)
        min_col = max(0, min(start_c, end_c) - extension)
        max_col = min(cols - 1, max(start_c, end_c) + extension)

        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if output_grid.get_cell(r, c) != 2 and is_within_diamond(r, c, line, extension):
                    output_grid.set_cell(r, c, 8)

    return output_grid
