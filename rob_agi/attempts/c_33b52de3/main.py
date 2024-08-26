from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Set

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Identify the "key" area (small colored pattern in any corner of the grid).
    2. Extract unique colors from the key pattern.
    3. Create a color distribution algorithm based on the key pattern.
    4. Map the gray patterns in the grid.
    5. Apply the color distribution to the gray areas.
    6. Preserve existing colored areas and pattern structure.
    
    The color distribution is applied across the grid, maintaining the 3x3 structure
    of the original gray blocks and preserving existing non-gray colors.
    """
    
    def find_key_area(grid: ColoredGrid) -> Tuple[List[List[int]], Tuple[int, int]]:
        rows, cols = grid.get_dimensions()
        corners = [(0, 0), (0, cols-7), (rows-7, 0), (rows-7, cols-7)]
        for top, left in corners:
            key_area = []
            for r in range(top, min(top+7, rows)):
                row = []
                for c in range(left, min(left+7, cols)):
                    if grid.get_cell(r, c) not in [0, 5]:
                        row.append(grid.get_cell(r, c))
                if row:
                    key_area.append(row)
            if key_area:
                return key_area, (top, left)
        return [], (0, 0)  # Return empty list and default position if no key found

    def get_unique_colors(key_area: List[List[int]]) -> Set[int]:
        return set(color for row in key_area for color in row if color not in [0, 5])

    def color_distribution(unique_colors: Set[int], r: int, c: int) -> int:
        colors = list(unique_colors)
        return colors[(r + c) % len(colors)]

    def apply_color_distribution(grid: ColoredGrid, unique_colors: Set[int], key_pos: Tuple[int, int]) -> ColoredGrid:
        new_grid = grid.deep_copy()
        rows, cols = new_grid.get_dimensions()
        start_row = key_pos[0] + 7 if key_pos[0] == 0 else 0
        start_col = key_pos[1] + 7 if key_pos[1] == 0 else 0
        
        for r in range(start_row, rows):
            for c in range(start_col, cols):
                if new_grid.get_cell(r, c) == 5:
                    new_color = color_distribution(unique_colors, r - start_row, c - start_col)
                    new_grid.set_cell(r, c, new_color)
        
        return new_grid

    key_area, key_pos = find_key_area(input_grid)
    unique_colors = get_unique_colors(key_area)
    output_grid = apply_color_distribution(input_grid, unique_colors, key_pos)

    return output_grid
