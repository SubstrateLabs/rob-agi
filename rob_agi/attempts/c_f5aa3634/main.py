from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f5aa3634(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by finding a repeating corner pattern.
    
    This function identifies a pattern in either the top-left or top-right corner
    of the input grid, searches for a matching pattern elsewhere in the grid,
    and returns the smallest subgrid containing the full pattern.
    
    Steps:
    1. Check the top-left corner for a pattern.
    2. If a match is found, return the top-left subgrid.
    3. If no match in top-left, check the top-right corner.
    4. If a match is found, return the top-right subgrid.
    5. If no match is found, return None.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid containing the repeating pattern,
                     or None if no pattern is found.
    """
    rows, cols = input_grid.get_dimensions()

    def get_corner_pattern(start_col: int, direction: int) -> List[Tuple[int, int]]:
        pattern = []
        for r in range(rows):
            for c in range(cols):
                actual_c = start_col + direction * c
                if 0 <= actual_c < cols:
                    if input_grid.values[r][actual_c] != 0:
                        pattern.append((r, actual_c))
                    elif pattern:
                        return pattern
        return pattern

    def find_match(pattern: List[Tuple[int, int]]) -> bool:
        if not pattern:
            return False
        pattern_height = max(r for r, _ in pattern) - min(r for r, _ in pattern) + 1
        pattern_width = max(c for _, c in pattern) - min(c for _, c in pattern) + 1
        
        for r in range(rows - pattern_height + 1):
            for c in range(cols - pattern_width + 1):
                if (r, c) != (0, 0) and (r, cols - pattern_width) != (0, cols - pattern_width):
                    if all(input_grid.values[r + dr][c + dc] == input_grid.values[pr][pc]
                           for (pr, pc), (dr, dc) in zip(pattern, [(r - pattern[0][0], c - pattern[0][1]) for _ in pattern])):
                        return True
        return False

    # Check top-left corner
    top_left_pattern = get_corner_pattern(0, 1)
    if find_match(top_left_pattern):
        pattern_height = max(r for r, _ in top_left_pattern) - min(r for r, _ in top_left_pattern) + 1
        pattern_width = max(c for _, c in top_left_pattern) - min(c for _, c in top_left_pattern) + 1
        return input_grid.extract_subgrid(0, 0, pattern_height, pattern_width)

    # Check top-right corner
    top_right_pattern = get_corner_pattern(cols - 1, -1)
    if find_match(top_right_pattern):
        pattern_height = max(r for r, _ in top_right_pattern) - min(r for r, _ in top_right_pattern) + 1
        pattern_width = max(c for _, c in top_right_pattern) - min(c for _, c in top_right_pattern) + 1
        return input_grid.extract_subgrid(0, cols - pattern_width, pattern_height, pattern_width)

    return None
