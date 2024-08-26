from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Expands green (3) squares vertically across the entire grid and horizontally within sections.
    2. Expands red (2) squares horizontally within sections and vertically upward.
    3. Creates magenta (6) squares where expanded red overlaps with expanded green in sections that originally contained green.
    4. Respects sky blue (8) grid lines as boundaries for horizontal expansion.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    sections = find_sections(output_grid)
    green_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    red_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    
    # Green Expansion
    for y in range(input_grid.num_rows):
        for x in range(input_grid.num_cols):
            if input_grid.values[y][x] == 3:
                expand_green(output_grid, green_mask, x, y, sections)
    
    # Red Expansion
    for section in sections:
        expand_red_in_section(input_grid, output_grid, red_mask, section)
    
    # Final Color Assignment and Magenta Creation
    assign_colors(input_grid, output_grid, green_mask, red_mask)
    
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

def expand_green(grid: ColoredGrid, green_mask: List[List[bool]], x: int, y: int, sections: List[Tuple[int, int, int, int]]):
    """Expands green vertically across the entire grid and horizontally within its section."""
    # Vertical expansion
    for r in range(grid.num_rows):
        green_mask[r][x] = True
    
    # Horizontal expansion within section
    section = next((s for s in sections if s[0] <= y <= s[2] and s[1] <= x <= s[3]), None)
    if section:
        _, left, _, right = section
        for c in range(left, right + 1):
            green_mask[y][c] = True

def expand_red_in_section(input_grid: ColoredGrid, output_grid: ColoredGrid, red_mask: List[List[bool]], section: Tuple[int, int, int, int]):
    """Expands red horizontally within its section and vertically upward."""
    top, left, bottom, right = section
    red_squares = [(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1) if input_grid.values[r][c] == 2]
    
    if len(red_squares) == 1:
        r, c = red_squares[0]
        for col in range(max(left, c - 1), min(right + 1, c + 2)):
            red_mask[r][col] = True
        for row in range(top, r + 1):
            red_mask[row][c] = True
    elif len(red_squares) > 1:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if input_grid.values[r][c] != 8:
                    red_mask[r][c] = True

def assign_colors(input_grid: ColoredGrid, output_grid: ColoredGrid, green_mask: List[List[bool]], red_mask: List[List[bool]]):
    """Assigns final colors based on the influence masks and creates magenta where appropriate."""
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if output_grid.values[r][c] == 8:
                continue
            elif red_mask[r][c] and green_mask[r][c]:
                if any(input_grid.values[rr][c] == 3 for rr in range(input_grid.num_rows)):
                    output_grid.values[r][c] = 6  # Magenta
                else:
                    output_grid.values[r][c] = 2  # Red
            elif red_mask[r][c]:
                output_grid.values[r][c] = 2  # Red
            elif green_mask[r][c]:
                output_grid.values[r][c] = 3  # Green
