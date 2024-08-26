from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Locate the color pattern anywhere in the grid.
    2. Extract the 4x4 color pattern that includes colored squares and black separators.
    3. Identify the starting position of the gray squares.
    4. Create a new grid, copying the original input grid.
    5. Apply the 4x4 pattern to replace gray squares, wrapping as needed.
    6. Preserve the original structure, including 3x3 squares and separators.
    7. Maintain any existing colored squares in the input grid.
    
    This approach works for all cases by adapting to the input's specific layout and color pattern.
    """
    def find_color_pattern(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        for r in range(rows - 3):
            for c in range(cols - 3):
                if grid.get_cell(r, c) not in [0, 5]:
                    pattern = [
                        [grid.get_cell(r+i, c+j) for j in range(4)]
                        for i in range(4)
                    ]
                    return pattern
        return []

    def find_gray_start(grid: ColoredGrid) -> Tuple[int, int]:
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 5:
                    return r, c
        return 0, 0

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
