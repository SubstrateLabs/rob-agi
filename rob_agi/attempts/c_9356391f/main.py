from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9356391f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a centered pattern based on unique colors from the top row.
    
    1. Extracts unique colors from the top row, preserving their order.
    2. Modifies the top row by replacing the rightmost occurrence of the last unique color with 5.
    3. Creates a square pattern using these colors in reverse order.
    4. Centers the pattern in the grid, preserving the top two rows.
    5. Fills the remaining space with 0s.
    """
    def find_unique_colors(row: List[int]) -> List[int]:
        return [c for c in row if c not in [0, 5] and c not in row[:row.index(c)]]

    rows, cols = input_grid.get_dimensions()
    unique_colors = find_unique_colors(input_grid.values[0])
    
    # Modify top row
    output_grid = input_grid.deep_copy()
    last_unique_color = unique_colors[-1]
    for c in range(cols-1, -1, -1):
        if output_grid.values[0][c] == last_unique_color:
            output_grid.values[0][c] = 5
            break
    
    # Create pattern
    pattern_size = min(rows - 2, cols - 2, 2 * len(unique_colors) - 1)
    if pattern_size % 2 == 0:
        pattern_size -= 1
    
    pattern = [[0 for _ in range(pattern_size)] for _ in range(pattern_size)]
    reversed_colors = list(reversed(unique_colors))
    
    for i in range(len(reversed_colors)):
        color = reversed_colors[i]
        start = i
        end = pattern_size - 1 - i
        for j in range(start, end + 1):
            pattern[start][j] = color
            pattern[end][j] = color
            pattern[j][start] = color
            pattern[j][end] = color
    
    # Position pattern in output grid
    start_row = 2
    start_col = (cols - pattern_size) // 2
    
    for r in range(pattern_size):
        for c in range(pattern_size):
            output_grid.values[start_row + r][start_col + c] = pattern[r][c]
    
    return output_grid
