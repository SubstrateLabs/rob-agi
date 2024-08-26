from rob_agi.colored_grid import ColoredGrid
from typing import List, Optional

def solve_33b52de3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray (5) patterns with colored patterns.
    
    The solution follows these steps:
    1. Search for a 4x4 color pattern anywhere in the grid.
    2. Create a new grid, copying the original input grid.
    3. Apply the 4x4 pattern to replace gray squares, wrapping as needed.
    4. Preserve the original structure, including 3x3 squares and separators.
    5. Maintain any existing colored squares in the input grid.
    
    This approach works for all cases by finding and applying the color pattern
    regardless of its location in the input grid.
    """
    def find_color_pattern(grid: ColoredGrid) -> Optional[List[List[int]]]:
        rows, cols = grid.get_dimensions()
        for r in range(rows - 3):
            for c in range(cols - 3):
                pattern = [
                    [grid.get_cell(r + i, c + j) for j in range(4)]
                    for i in range(4)
                ]
                if all(cell != 0 and cell != 5 for row in pattern for cell in row):
                    return pattern
        return None

    def apply_pattern(grid: ColoredGrid, pattern: List[List[int]]) -> ColoredGrid:
        new_grid = grid.deep_copy()
        rows, cols = new_grid.get_dimensions()
        pattern_rows, pattern_cols = len(pattern), len(pattern[0])
        
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) == 5:
                    pattern_r, pattern_c = r % pattern_rows, c % pattern_cols
                    new_color = pattern[pattern_r][pattern_c]
                    new_grid.set_cell(r, c, new_color)
        
        return new_grid

    color_pattern = find_color_pattern(input_grid)
    if color_pattern is None:
        return input_grid  # Return original grid if no pattern found
    output_grid = apply_pattern(input_grid, color_pattern)

    return output_grid
