from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-8 pixels (inclusive) are changed to red or green.
    2. The target color (red or green) is determined by the global prevalence of these colors.
    3. Larger blue regions, single blue pixels, and other colors remain unchanged.
    4. All transformations are applied simultaneously.
    5. Gray (5) acts as a border and is not considered part of any region.
    6. Only orthogonally adjacent cells are considered part of the same region.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5
    rows, cols = input_grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(start_row: int, start_col: int) -> Tuple[int, List[Tuple[int, int]]]:
        stack = [(start_row, start_col)]
        region = []
        while stack:
            r, c = stack.pop()
            if (0 <= r < rows and 0 <= c < cols and
                not visited[r][c] and input_grid.values[r][c] == BLUE):
                visited[r][c] = True
                region.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))
        return len(region), region

    # Determine target color
    red_count = sum(row.count(RED) for row in input_grid.values)
    green_count = sum(row.count(GREEN) for row in input_grid.values)
    target_color = RED if red_count >= green_count else GREEN

    # Identify regions to transform
    regions_to_transform = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == BLUE and not visited[r][c]:
                size, region = flood_fill(r, c)
                if 2 <= size <= 8:
                    regions_to_transform.append(region)

    # Apply transformations
    result_grid = input_grid.deep_copy()
    for region in regions_to_transform:
        for r, c in region:
            result_grid.values[r][c] = target_color

    return result_grid
