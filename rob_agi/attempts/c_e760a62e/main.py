from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Expands green (3) squares horizontally and vertically across the entire grid, stopping only at sky blue (8) lines.
    2. Expands red (2) squares horizontally within sections and vertically upward, overwriting green.
    3. Creates magenta (6) squares where expanded red overlaps with expanded green in sections that originally contained green.
    4. Respects sky blue (8) grid lines as boundaries throughout the process.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    sections = find_sections(output_grid)
    
    # Green Expansion
    for y in range(output_grid.num_rows):
        for x in range(output_grid.num_cols):
            if input_grid.values[y][x] == 3:
                expand_green(output_grid, x, y)
    
    # Red Expansion
    for y in range(output_grid.num_rows):
        for x in range(output_grid.num_cols):
            if input_grid.values[y][x] == 2:
                expand_red(output_grid, sections, x, y)
    
    # Magenta Creation
    create_magenta(input_grid, output_grid, sections)
    
    return output_grid

def find_sections(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """Identifies and returns section boundaries defined by sky blue (8) lines."""
    rows, cols = grid.get_dimensions()
    sections = []
    start_row, start_col = 0, 0

    for r in range(rows + 1):
        if r == rows or all(grid.values[r][c] == 8 for c in range(cols)):
            if start_row < r:
                for c in range(cols + 1):
                    if c == cols or grid.values[start_row][c] == 8:
                        if start_col < c:
                            sections.append((start_row, start_col, r - 1, c - 1))
                        start_col = c + 1
            start_row = r + 1
            start_col = 0

    return sections

def expand_green(grid: ColoredGrid, x: int, y: int):
    """Expands green horizontally and vertically across the entire grid, stopping at sky blue lines."""
    # Horizontal expansion
    for c in range(grid.num_cols):
        if grid.values[y][c] != 8:
            grid.values[y][c] = 3
    
    # Vertical expansion
    for r in range(grid.num_rows):
        if grid.values[r][x] != 8:
            grid.values[r][x] = 3

def expand_red(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], x: int, y: int):
    """Expands red horizontally within its section and vertically upward, overwriting green."""
    section = next((s for s in sections if s[0] <= y <= s[2] and s[1] <= x <= s[3]), None)
    if section:
        top, left, bottom, right = section
        # Horizontal expansion within section
        for c in range(left, right + 1):
            grid.values[y][c] = 2
        # Vertical expansion upward
        for r in range(top, y + 1):
            if grid.values[r][x] != 8:
                grid.values[r][x] = 2

def create_magenta(input_grid: ColoredGrid, output_grid: ColoredGrid, sections: List[Tuple[int, int, int, int]]):
    """Creates magenta where expanded red overlaps with expanded green in sections that originally contained green."""
    for section in sections:
        top, left, bottom, right = section
        if any(input_grid.values[r][c] == 3 for r in range(top, bottom + 1) for c in range(left, right + 1)):
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if output_grid.values[r][c] == 2 and any(input_grid.values[rr][c] == 3 for rr in range(top, bottom + 1)):
                        output_grid.values[r][c] = 6
