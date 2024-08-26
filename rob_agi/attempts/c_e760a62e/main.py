from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Creates "zone of influence" masks for red and green squares.
    2. Expands green (3) squares vertically across the entire grid and horizontally within sections.
    3. Expands red (2) squares horizontally within sections and vertically upward.
    4. Creates magenta (6) squares where expanded red overlaps with expanded green in sections that originally contained green or were empty.
    5. Respects sky blue (8) grid lines as boundaries for expansion.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    sections = find_sections(output_grid)
    green_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    red_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    
    # Create zone of influence masks
    create_influence_masks(input_grid, green_mask, red_mask, sections)
    
    # Apply expansions and assign colors
    assign_colors(input_grid, output_grid, green_mask, red_mask, sections)
    
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

def create_influence_masks(input_grid: ColoredGrid, green_mask: List[List[bool]], red_mask: List[List[bool]], sections: List[Tuple[int, int, int, int]]):
    """Creates zone of influence masks for red and green squares."""
    for section in sections:
        top, left, bottom, right = section
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if input_grid.values[r][c] == 2:  # Red
                    for rr in range(top, r + 1):
                        for cc in range(left, right + 1):
                            red_mask[rr][cc] = True
                elif input_grid.values[r][c] == 3:  # Green
                    for rr in range(input_grid.num_rows):
                        green_mask[rr][c] = True
                    for cc in range(left, right + 1):
                        green_mask[r][cc] = True

def assign_colors(input_grid: ColoredGrid, output_grid: ColoredGrid, green_mask: List[List[bool]], red_mask: List[List[bool]], sections: List[Tuple[int, int, int, int]]):
    """Assigns final colors based on the influence masks and creates magenta where appropriate."""
    for section in sections:
        top, left, bottom, right = section
        original_colors = set(input_grid.values[r][c] for r in range(top, bottom + 1) for c in range(left, right + 1))
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if output_grid.values[r][c] == 8:
                    continue
                elif green_mask[r][c] and red_mask[r][c]:
                    if 3 in original_colors or (2 not in original_colors and 3 not in original_colors):
                        output_grid.values[r][c] = 6  # Magenta
                    else:
                        output_grid.values[r][c] = 2  # Red
                elif green_mask[r][c]:
                    output_grid.values[r][c] = 3  # Green
                elif red_mask[r][c]:
                    output_grid.values[r][c] = 2  # Red
