from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ef26cbf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies yellow (4) lines that divide the grid into rows of sections.
    2. For each row of sections:
       - Finds the leftmost non-zero, non-yellow color.
       - Applies this color to all non-zero, non-yellow cells in the row's sections.
    3. Preserves yellow lines and originally empty (black) cells.
    4. Maintains the original pattern of filled and empty spaces in each section.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    yellow_lines = find_yellow_lines(grid)
    rows = get_rows(grid, yellow_lines)
    
    for row in rows:
        leftmost_color = find_leftmost_color(grid, row)
        if leftmost_color:
            apply_color_to_row(grid, row, leftmost_color)
    
    return grid

def find_yellow_lines(grid: ColoredGrid) -> List[int]:
    return [r for r, row in enumerate(grid.values) if all(cell == 4 for cell in row)]

def get_rows(grid: ColoredGrid, yellow_lines: List[int]) -> List[Tuple[int, int]]:
    rows = []
    start = 0
    for line in yellow_lines + [len(grid.values)]:
        if line > start:
            rows.append((start, line))
        start = line + 1
    return rows

def find_leftmost_color(grid: ColoredGrid, row: Tuple[int, int]) -> int:
    start, end = row
    for r in range(start, end):
        for cell in grid.values[r]:
            if cell not in [0, 4]:
                return cell
    return 0

def apply_color_to_row(grid: ColoredGrid, row: Tuple[int, int], color: int):
    start, end = row
    for r in range(start, end):
        for c, cell in enumerate(grid.values[r]):
            if cell not in [0, 4]:
                grid.values[r][c] = color
