from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. All connected blue regions (including single cells) are changed to red or green.
    2. The target color (red or green) is determined by the presence of that color in the grid.
    3. If neither red nor green is present, green is used as the target color.
    4. Gray areas and other colors remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Determine target color (red if it exists, otherwise green)
    target_color = 2 if any(2 in row for row in output_grid.values) else 3

    def find_connected_region(row: int, col: int) -> List[Tuple[int, int]]:
        region = []
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] == 1 and (r, c) not in region:
                region.append((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return region

    # Find and transform all blue regions
    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1:
                region = find_connected_region(row, col)
                for r, c in region:
                    output_grid.values[r][c] = target_color

    return output_grid
