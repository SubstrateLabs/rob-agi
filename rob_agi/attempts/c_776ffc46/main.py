from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 3x3 or smaller are changed to red or green.
    2. The target color (red or green) is determined by the global prevalence of these colors.
    3. Larger blue regions remain unchanged.
    4. Single blue pixels and other colors (including gray) remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_connected_region(row: int, col: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        region = []
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] == color and (r, c) not in visited:
                region.append((r, c))
                visited.add((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return region

    def get_region_dimensions(region: List[Tuple[int, int]]) -> Tuple[int, int]:
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        return max_row - min_row + 1, max_col - min_col + 1

    # Determine global color prevalence
    red_count = sum(row.count(2) for row in output_grid.values)
    green_count = sum(row.count(3) for row in output_grid.values)
    preferred_color = 2 if red_count >= green_count else 3

    visited = set()
    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1 and (row, col) not in visited:
                region = find_connected_region(row, col, 1, visited)
                height, width = get_region_dimensions(region)
                
                if 1 < len(region) <= 9 and height <= 3 and width <= 3:
                    for r, c in region:
                        output_grid.values[r][c] = preferred_color

    return output_grid
