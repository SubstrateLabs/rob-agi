from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-5 pixels are changed to red or green.
    2. Blue regions of size 6-8 pixels are changed only if they form a plus shape.
    3. The target color (red or green) is determined by the global prevalence of these colors.
    4. Larger blue regions, single blue pixels, and other colors remain unchanged.
    5. All transformations are applied simultaneously.
    6. Gray (5) acts as a border and is not considered part of any region.
    7. Only orthogonally adjacent cells are considered part of the same region.
    """
    BLUE, RED, GREEN = 1, 2, 3

    def flood_fill(grid, x, y):
        region = []
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) not in region and 0 <= cx < len(grid) and 0 <= cy < len(grid[0]) and grid[cx][cy] == BLUE:
                region.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))
        return region

    def is_plus_shape(region):
        if len(region) not in [5, 7]:  # Plus shapes can only have 5 or 7 cells
            return False
        center = region[len(region) // 2]
        arms = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return all((center[0] + dx, center[1] + dy) in region for dx, dy in arms)

    def apply_transformations(grid, regions_to_transform, target_color):
        new_grid = grid.deep_copy()
        for region in regions_to_transform:
            for x, y in region:
                new_grid.values[x][y] = target_color
        return new_grid

    red_count = sum(row.count(RED) for row in input_grid.values)
    green_count = sum(row.count(GREEN) for row in input_grid.values)
    target_color = RED if red_count >= green_count else GREEN

    blue_regions = []
    visited = set()
    for i in range(len(input_grid.values)):
        for j in range(len(input_grid.values[0])):
            if input_grid.values[i][j] == BLUE and (i, j) not in visited:
                region = flood_fill(input_grid.values, i, j)
                visited.update(region)
                blue_regions.append(region)

    regions_to_transform = []
    for region in blue_regions:
        size = len(region)
        if 2 <= size <= 5 or (6 <= size <= 8 and is_plus_shape(region)):
            regions_to_transform.append(region)

    return apply_transformations(input_grid, regions_to_transform, target_color)
