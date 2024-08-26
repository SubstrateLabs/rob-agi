from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Identify the "key" area (small colored pattern in any corner of the grid).
    2. Generate a coloring sequence based on the key area.
    3. Map the gray patterns in the grid.
    4. Apply the coloring sequence to the gray patterns.
    5. Preserve existing colored areas and pattern structure.
    
    If no key is found, a default color sequence is used.
    The coloring sequence is applied to patterns, repeating if necessary.
    The structure of each pattern is maintained, only changing gray (5) to the new color.
    """
    
    def find_key_area(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        corners = [(0, 0), (0, cols-5), (rows-5, 0), (rows-5, cols-5)]
        for top, left in corners:
            key_area = []
            for r in range(top, top+5):
                row = []
                for c in range(left, left+5):
                    if grid.get_cell(r, c) not in [0, 5]:
                        row.append(grid.get_cell(r, c))
                if row:
                    key_area.append(row)
            if key_area:
                return key_area
        return []  # Return empty list if no key found

    def generate_color_sequence(key_area: List[List[int]]) -> List[int]:
        if not key_area:
            return [1, 2, 3, 4]  # Default sequence if no key found
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
        if not color_sequence:
            return grid  # Return original grid if color sequence is empty
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
