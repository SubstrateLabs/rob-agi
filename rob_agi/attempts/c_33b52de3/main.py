from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Identify the "key" area (small colored pattern in any corner of the grid).
    2. Extract and expand the key pattern.
    3. Map the gray patterns in the grid.
    4. Apply the expanded pattern to the gray areas.
    5. Preserve existing colored areas and pattern structure.
    
    The expanded pattern is applied across the grid, maintaining the 3x3 structure
    of the original gray blocks and preserving existing non-gray colors.
    """
    
    def find_key_area(grid: ColoredGrid) -> Tuple[List[List[int]], Tuple[int, int]]:
        rows, cols = grid.get_dimensions()
        corners = [(0, 0), (0, cols-6), (rows-6, 0), (rows-6, cols-6)]
        for top, left in corners:
            key_area = []
            for r in range(top, min(top+6, rows)):
                row = []
                for c in range(left, min(left+6, cols)):
                    if grid.get_cell(r, c) not in [0, 5]:
                        row.append(grid.get_cell(r, c))
                if row:
                    key_area.append(row)
            if key_area:
                return key_area, (top, left)
        return [], (0, 0)  # Return empty list and default position if no key found

    def expand_key_pattern(key_area: List[List[int]]) -> List[List[int]]:
        if not key_area:
            return [[1, 2], [3, 4]]  # Default pattern if no key found
        
        # Expand horizontally
        expanded_h = [row * 2 for row in key_area]
        
        # Expand vertically
        expanded_v = expanded_h * 2
        
        return expanded_v

    def apply_expanded_pattern(grid: ColoredGrid, expanded_pattern: List[List[int]], key_pos: Tuple[int, int]) -> ColoredGrid:
        new_grid = grid.deep_copy()
        rows, cols = new_grid.get_dimensions()
        pattern_rows, pattern_cols = len(expanded_pattern), len(expanded_pattern[0])
        
        start_row = key_pos[0] + 6 if key_pos[0] == 0 else 0
        start_col = key_pos[1] + 6 if key_pos[1] == 0 else 0
        
        for r in range(start_row, rows):
            for c in range(start_col, cols):
                if new_grid.get_cell(r, c) == 5:
                    pattern_r, pattern_c = (r - start_row) % pattern_rows, (c - start_col) % pattern_cols
                    new_grid.set_cell(r, c, expanded_pattern[pattern_r][pattern_c])
        
        return new_grid

    key_area, key_pos = find_key_area(input_grid)
    expanded_pattern = expand_key_pattern(key_area)
    output_grid = apply_expanded_pattern(input_grid, expanded_pattern, key_pos)

    return output_grid
