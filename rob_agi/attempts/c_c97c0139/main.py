from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c97c0139(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c97c0139 challenge by creating diamond-shaped sky blue (8) fields around red (2) lines.
    
    The solution follows these steps:
    1. Identify red lines in the input grid.
    2. For each red line, create a diamond-shaped field around it.
    3. Set non-red cells within the diamond to sky blue.
    
    This approach works for both horizontal and vertical red lines of varying lengths.
    The diamond shape extends outwards from the red line, with its size determined by
    half the length of the red line.
    """
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

    def calculate_distance(row: int, col: int, line: Tuple[int, int, int, int]) -> int:
        start_r, start_c, end_r, end_c = line
        if start_r == end_r:  # Horizontal line
            return abs(row - start_r) + min(abs(col - start_c), abs(col - end_c))
        else:  # Vertical line
            return abs(col - start_c) + min(abs(row - start_r), abs(row - end_r))

    def is_within_diamond(row: int, col: int, line: Tuple[int, int, int, int]) -> bool:
        start_r, start_c, end_r, end_c = line
        line_length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
        return calculate_distance(row, col, line) < line_length // 2 + 1

    # Find red lines
    red_lines = find_red_lines(input_grid)

    # Create output grid
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Process each cell
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) != 2:  # If not red
                for line in red_lines:
                    if is_within_diamond(r, c, line):
                        output_grid.set_cell(r, c, 8)  # Set to sky blue
                        break

    return output_grid
