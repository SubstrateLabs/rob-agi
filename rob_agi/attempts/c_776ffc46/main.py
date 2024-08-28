from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-8 pixels (inclusive) are changed to red or green.
    2. The target color (red or green) is determined by the global prevalence of these colors.
    3. Larger blue regions, single blue pixels, and other colors remain unchanged.
    4. All transformations are applied simultaneously.
    5. Gray (5) acts as a border and is not considered part of any region.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_connected_region(row: int, col: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        region = []
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (0 <= r < rows and 0 <= c < cols and 
                output_grid.values[r][c] == color and 
                (r, c) not in visited and
                output_grid.values[r][c] != 5):  # Don't cross gray borders
                region.append((r, c))
                visited.add((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return region

    def count_color_prevalence():
        red_count = sum(row.count(2) for row in input_grid.values)
        green_count = sum(row.count(3) for row in input_grid.values)
        return 2 if red_count >= green_count else 3

    def is_transformable(region: List[Tuple[int, int]]) -> bool:
        return 2 <= len(region) <= 8

    target_color = count_color_prevalence()
    visited = set()
    transform_coords = set()

    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1 and (row, col) not in visited:
                region = find_connected_region(row, col, 1, visited)
                if is_transformable(region):
                    transform_coords.update(region)

    for row, col in transform_coords:
        output_grid.values[row][col] = target_color

    return output_grid
