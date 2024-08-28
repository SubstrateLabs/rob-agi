from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-9 pixels are changed to red or green if:
       - They form a line (horizontal or vertical) of length <= 9, or
       - Their bounding box has both dimensions <= 3
    2. The target color (red or green) is determined by the global prevalence of these colors.
    3. Larger blue regions, single blue pixels, and other colors remain unchanged.
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

    def is_line(region: List[Tuple[int, int]]) -> bool:
        return len(set(r for r, _ in region)) == 1 or len(set(c for _, c in region)) == 1

    def get_dimensions(region: List[Tuple[int, int]]) -> Tuple[int, int]:
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        return max_row - min_row + 1, max_col - min_col + 1

    def should_transform_region(region: List[Tuple[int, int]]) -> bool:
        if len(region) < 2 or len(region) > 9:
            return False
        if is_line(region):
            return True
        height, width = get_dimensions(region)
        return height <= 3 and width <= 3

    # Determine global color prevalence
    total_pixels = rows * cols
    red_count = sum(row.count(2) for row in output_grid.values)
    green_count = sum(row.count(3) for row in output_grid.values)
    preferred_color = 2 if red_count >= green_count else 3

    visited = set()
    transform_coords = set()

    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1 and (row, col) not in visited:
                region = find_connected_region(row, col, 1, visited)
                if should_transform_region(region):
                    transform_coords.update(region)

    for row in range(rows):
        for col in range(cols):
            if (row, col) in transform_coords:
                output_grid.values[row][col] = preferred_color

    return output_grid
