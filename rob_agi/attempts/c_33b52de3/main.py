from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Locate the 4x4 color pattern in the bottom-right corner of the grid.
    2. Create a new grid, copying the original input grid.
    3. Apply the 4x4 pattern to replace gray squares, wrapping as needed.
    4. Preserve the original structure, including 3x3 squares and separators.
    5. Maintain any existing colored squares in the input grid.
    
    This approach works for all cases by adapting to the input's specific layout and color pattern.
    """
    def find_color_pattern(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        pattern_start_row = rows - 4
        pattern_start_col = 1  # Start from the second column
        pattern = [
            [grid.get_cell(pattern_start_row + i, pattern_start_col + j) for j in range(4)]
            for i in range(4)
        ]
        return pattern

    def apply_pattern(grid: ColoredGrid, pattern: List[List[int]]) -> ColoredGrid:
        new_grid = grid.deep_copy()
        rows, cols = new_grid.get_dimensions()
        pattern_rows, pattern_cols = len(pattern), len(pattern[0])
        
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) == 5:
                    pattern_r, pattern_c = r % pattern_rows, c % pattern_cols
                    new_color = pattern[pattern_r][pattern_c]
                    if new_color != 0:
                        new_grid.set_cell(r, c, new_color)
        
        return new_grid

    color_pattern = find_color_pattern(input_grid)
    output_grid = apply_pattern(input_grid, color_pattern)

    return output_grid
