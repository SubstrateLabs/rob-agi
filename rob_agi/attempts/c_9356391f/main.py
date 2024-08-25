from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9356391f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a centered pattern based on unique colors from the top row.
    
    1. Extracts unique colors from the top row, preserving their order.
    2. Modifies the top row by replacing the rightmost occurrence of the last unique color with 5.
    3. Determines the maximum pattern size that fits in the grid.
    4. Creates a concentric square pattern using these colors in reverse order.
    5. Centers the pattern in the grid, preserving the second row of 5s.
    6. Fills the remaining space with 0s.
    """
    def find_unique_colors(row: List[int]) -> List[int]:
        return [c for c in row if c not in [0, 5] and c not in row[:row.index(c)]]

    def create_pattern(colors: List[int], size: int) -> List[List[int]]:
        pattern = [[0 for _ in range(size)] for _ in range(size)]
        for i in range((size + 1) // 2):
            color = colors[i % len(colors)]
            for r in range(i, size-i):
                for c in range(i, size-i):
                    if r == i or r == size-i-1 or c == i or c == size-i-1:
                        pattern[r][c] = color
        return pattern

    rows, cols = input_grid.get_dimensions()
    unique_colors = find_unique_colors(input_grid.values[0])
    
    # Modify top row
    output_grid = input_grid.deep_copy()
    last_unique_color = unique_colors[-1]
    for c in range(cols-1, -1, -1):
        if output_grid.values[0][c] == last_unique_color:
            output_grid.values[0][c] = 5
            break
    
    # Determine pattern size
    pattern_size = min(max(row for row in range(1, rows-1, 2)),
                       max(col for col in range(1, cols, 2)))
    
    pattern = create_pattern(list(reversed(unique_colors)), pattern_size)
    
    # Center and apply pattern
    start_col = (cols - pattern_size) // 2
    for r in range(pattern_size):
        for c in range(pattern_size):
            output_grid.values[r + 2][c + start_col] = pattern[r][c]
    
    return output_grid
