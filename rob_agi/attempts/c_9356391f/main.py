from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9356391f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern based on unique colors from the top row.
    
    1. Extracts unique colors from the top row, preserving their order.
    2. Determines pattern size based on the number of unique colors and grid dimensions.
    3. Creates a concentric square pattern using these colors in reverse order.
    4. Applies the pattern to the bottom-left of the grid, preserving the second row of 5s.
    5. Adjusts the top row by replacing the rightmost duplicate used color with 5.
    """
    def find_unique_colors(row: List[int]) -> List[int]:
        return [c for c in row if c not in [0, 5] and c not in row[:row.index(c)]]

    def create_pattern(colors: List[int], size: int) -> List[List[int]]:
        pattern = [[0 for _ in range(size)] for _ in range(size)]
        for i, color in enumerate(reversed(colors)):
            for r in range(i, size-i):
                for c in range(i, size-i):
                    if r == i or r == size-i-1 or c == i or c == size-i-1:
                        pattern[r][c] = color
        return pattern

    unique_colors = find_unique_colors(input_grid.values[0])
    pattern_size = max(2 * len(unique_colors) + 1, len(input_grid.values) - 2)
    pattern = create_pattern(unique_colors, pattern_size)
    
    output_grid = input_grid.deep_copy()
    
    # Apply pattern
    for r in range(pattern_size):
        for c in range(pattern_size):
            if r + 2 < len(output_grid.values) and c < len(output_grid.values[0]):
                output_grid.values[r + 2][c] = pattern[r][c]

    # Adjust top row
    used_colors = set()
    last_duplicate = None
    for c in range(len(output_grid.values[0])):
        if output_grid.values[0][c] in unique_colors:
            if output_grid.values[0][c] in used_colors:
                last_duplicate = c
            used_colors.add(output_grid.values[0][c])
    
    if last_duplicate is not None:
        output_grid.values[0][last_duplicate] = 5

    return output_grid
