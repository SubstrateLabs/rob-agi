from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Identify the "key" area (small colored pattern in the bottom-left corner).
    2. Generate a coloring sequence based on the key area.
    3. Map the gray patterns in the grid.
    4. Apply the coloring sequence to the gray patterns.
    5. Preserve existing colored areas.
    
    The coloring sequence is applied row by row, repeating if necessary.
    The structure of each pattern is maintained, only changing gray (5) to the new color.
    """
    
    def find_key_area(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        key_area = []
        for r in range(rows-3, rows):
            row = []
            for c in range(1, 6):
                if grid.get_cell(r, c) != 0:
                    row.append(grid.get_cell(r, c))
            if row:
                key_area.append(row)
        return key_area

    def generate_color_sequence(key_area: List[List[int]]) -> List[int]:
        return [color for row in key_area for color in row if color != 0]

    def map_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
        rows, cols = grid.get_dimensions()
        patterns = []
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 5:
                    if not patterns or r > patterns[-1][2] or c > patterns[-1][3]:
                        patterns.append((r, c, r+2, c+2))
        return patterns

    def apply_color_sequence(grid: ColoredGrid, patterns: List[Tuple[int, int, int, int]], color_sequence: List[int]) -> ColoredGrid:
        new_grid = grid.deep_copy()
        for i, (top, left, bottom, right) in enumerate(patterns):
            color = color_sequence[i % len(color_sequence)]
            for r in range(top, bottom+1):
                for c in range(left, right+1):
                    if grid.get_cell(r, c) == 5:
                        new_grid.set_cell(r, c, color)
            # Color the single square above the pattern, if it exists
            if top > 0 and grid.get_cell(top-1, left+1) == 5:
                new_grid.set_cell(top-1, left+1, color)
        return new_grid

    key_area = find_key_area(input_grid)
    color_sequence = generate_color_sequence(key_area)
    patterns = map_patterns(input_grid)
    output_grid = apply_color_sequence(input_grid, patterns, color_sequence)

    return output_grid
