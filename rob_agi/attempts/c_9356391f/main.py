from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9356391f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a centered pattern based on unique colors.
    
    1. Extracts unique colors from the top row.
    2. Determines pattern size and position based on these colors and an isolated color.
    3. Creates a concentric square pattern using these colors.
    4. Applies the pattern to the grid, preserving the second row of 5s.
    5. Adjusts the top row by replacing used colors with 5s.
    """
    def find_unique_colors(row: List[int]) -> List[int]:
        return sorted(set(c for c in row if c not in [0, 5]), reverse=True)

    def find_isolated_color(grid: List[List[int]]) -> Tuple[int, int, int]:
        for r in range(2, len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c] != 0:
                    return grid[r][c], r, c
        return 0, 0, 0

    def create_pattern(colors: List[int], size: int) -> List[List[int]]:
        pattern = [[0 for _ in range(size)] for _ in range(size)]
        for i, color in enumerate(colors):
            for r in range(i, size-i):
                for c in range(i, size-i):
                    if r == i or r == size-i-1 or c == i or c == size-i-1:
                        pattern[r][c] = color
        return pattern

    unique_colors = find_unique_colors(input_grid.values[0])
    isolated_color, isolated_r, isolated_c = find_isolated_color(input_grid.values)
    
    pattern_size = 2 * len(unique_colors) + 1
    pattern_top = max(2, isolated_r - pattern_size + 1)
    pattern_left = max(0, isolated_c - pattern_size + 1)

    pattern = create_pattern(unique_colors, pattern_size)
    
    output_grid = input_grid.deep_copy()
    
    # Apply pattern
    for r in range(pattern_size):
        for c in range(pattern_size):
            if pattern_top + r < len(output_grid.values) and pattern_left + c < len(output_grid.values[0]):
                if output_grid.values[pattern_top + r][pattern_left + c] != 5:
                    output_grid.values[pattern_top + r][pattern_left + c] = pattern[r][c]

    # Adjust top row
    used_colors = set()
    for c in range(len(output_grid.values[0])):
        if output_grid.values[0][c] in unique_colors:
            if output_grid.values[0][c] in used_colors:
                output_grid.values[0][c] = 5
            else:
                used_colors.add(output_grid.values[0][c])

    return output_grid
