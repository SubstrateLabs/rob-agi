from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ef26cbf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies yellow (4) lines that divide the grid into sections.
    2. For each section:
       - For each column in the section:
         * Finds the topmost non-zero, non-yellow color.
         * Propagates this color downwards in the column within the section.
    3. Preserves yellow lines and originally empty (black) cells.
    4. Maintains the original pattern of filled and empty spaces in each section.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    yellow_lines = find_yellow_lines(grid)
    sections = get_sections(grid, yellow_lines)
    
    for section in sections:
        process_section(grid, section)
    
    return grid

def find_yellow_lines(grid: ColoredGrid) -> List[int]:
    return [r for r, row in enumerate(grid.values) if all(cell == 4 for cell in row)]

def get_sections(grid: ColoredGrid, yellow_lines: List[int]) -> List[Tuple[int, int]]:
    sections = []
    start = 0
    for line in yellow_lines + [len(grid.values)]:
        if line > start:
            sections.append((start, line))
        start = line + 1
    return sections

def process_section(grid: ColoredGrid, section: Tuple[int, int]):
    start, end = section
    for col in range(len(grid.values[0])):
        top_color = find_top_color(grid, section, col)
        if top_color:
            propagate_color(grid, section, col, top_color)

def find_top_color(grid: ColoredGrid, section: Tuple[int, int], col: int) -> int:
    start, end = section
    for r in range(start, end):
        if grid.values[r][col] not in [0, 4]:
            return grid.values[r][col]
    return 0

def propagate_color(grid: ColoredGrid, section: Tuple[int, int], col: int, color: int):
    start, end = section
    for r in range(start, end):
        if grid.values[r][col] not in [0, 4]:
            grid.values[r][col] = color
