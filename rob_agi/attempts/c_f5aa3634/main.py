from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple

def solve_f5aa3634(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by finding the smallest repeating pattern.
    
    This function searches the entire grid for the smallest pattern that repeats
    at least once elsewhere in the grid. The pattern can be located anywhere and
    may appear more than twice.
    
    Steps:
    1. Iterate through each cell in the grid.
    2. For each non-black cell, expand to find a potential pattern.
    3. Search for a match of this pattern elsewhere in the grid.
    4. Keep track of the smallest pattern that has a match.
    5. Return the smallest repeating pattern found, or None if no pattern is found.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid containing the smallest repeating pattern,
                     or None if no pattern is found.
    """
    rows, cols = input_grid.get_dimensions()
    best_pattern = None
    best_pattern_size = float('inf')

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def expand_pattern(top: int, left: int, bottom: int, right: int) -> Tuple[int, int, int, int]:
        while True:
            expanded = False
            if top > 0 and any(input_grid.values[top-1][c] != 0 for c in range(left, right+1)):
                top -= 1
                expanded = True
            if bottom < rows-1 and any(input_grid.values[bottom+1][c] != 0 for c in range(left, right+1)):
                bottom += 1
                expanded = True
            if left > 0 and any(input_grid.values[r][left-1] != 0 for r in range(top, bottom+1)):
                left -= 1
                expanded = True
            if right < cols-1 and any(input_grid.values[r][right+1] != 0 for r in range(top, bottom+1)):
                right += 1
                expanded = True
            if not expanded:
                break
        return top, left, bottom, right

    def match_pattern(pattern: ColoredGrid, sr: int, sc: int) -> bool:
        pattern_rows, pattern_cols = pattern.get_dimensions()
        return all(
            input_grid.values[sr+r][sc+c] == pattern.values[r][c]
            for r in range(pattern_rows)
            for c in range(pattern_cols)
        )

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                top, left, bottom, right = expand_pattern(r, c, r, c)
                pattern_height = bottom - top + 1
                pattern_width = right - left + 1
                pattern_size = pattern_height * pattern_width

                if pattern_size < best_pattern_size:
                    pattern = input_grid.extract_subgrid(top, left, pattern_height, pattern_width)
                    
                    for sr in range(rows - pattern_height + 1):
                        for sc in range(cols - pattern_width + 1):
                            if (sr, sc) != (top, left) and match_pattern(pattern, sr, sc):
                                best_pattern = pattern
                                best_pattern_size = pattern_size
                                break
                        if best_pattern is not None:
                            break

    return best_pattern
