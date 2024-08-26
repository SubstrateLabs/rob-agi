from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Identifies sections bounded by sky blue (8) lines.
    2. Creates influence masks for red (2) and green (3) squares.
    3. Expands red horizontally within sections and vertically upward.
    4. Expands green vertically across the entire grid and horizontally within sections.
    5. Creates magenta (6) where red and green overlap.
    6. Preserves original structure and colors of non-expanding squares.

    The expansion follows these rules:
    - Red expands horizontally within its section and upwards to the top of its section.
    - Green expands vertically across the entire grid and horizontally within its section.
    - Where red and green expansions overlap, magenta (6) is created.
    - Sky blue lines and original colored squares are preserved.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    sections = find_sections(input_grid)
    green_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    red_mask = [[False for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)]
    
    create_influence_masks(input_grid, green_mask, red_mask, sections)
    apply_color_expansions(input_grid, output_grid, green_mask, red_mask)
    
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
    """Creates influence masks for red and green squares."""
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

def apply_color_expansions(input_grid: ColoredGrid, output_grid: ColoredGrid, green_mask: List[List[bool]], red_mask: List[List[bool]]):
    """Applies color expansions based on influence masks."""
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] == 8:
                continue
            elif input_grid.values[r][c] in {2, 3}:
                output_grid.values[r][c] = input_grid.values[r][c]
            elif red_mask[r][c] and green_mask[r][c]:
                output_grid.values[r][c] = 6
            elif red_mask[r][c]:
                output_grid.values[r][c] = 2
            elif green_mask[r][c]:
                output_grid.values[r][c] = 3
