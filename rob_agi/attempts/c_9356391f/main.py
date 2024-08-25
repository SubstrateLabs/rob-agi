from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9356391f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a centered pattern based on unique colors from the top row.
    
    1. Extracts unique colors from the top row, preserving their order.
    2. Modifies the top row by replacing the rightmost occurrence of the last unique color with 5.
    3. Creates a large rectangular pattern using these colors in reverse order.
    4. Centers the pattern in the grid, preserving the second row of 5s.
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
    pattern_height = rows - 2
    pattern_width = cols - 2
    reversed_colors = list(reversed(unique_colors))
    
    for r in range(2, rows):
        for c in range(1, cols-1):
            color_index = min(
                r - 2,
                c - 1,
                rows - r - 1,
                cols - c - 2,
                len(reversed_colors) - 1
            )
            output_grid.values[r][c] = reversed_colors[color_index]
    
    return output_grid
