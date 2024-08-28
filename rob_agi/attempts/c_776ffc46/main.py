from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

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
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5

    def flood_fill(grid: List[List[int]], x: int, y: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) not in region and 0 <= cx < len(grid) and 0 <= cy < len(grid[0]) and grid[cx][cy] == BLUE:
                region.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != GRAY:
                        queue.append((nx, ny))
        return region

    def is_plus_shape(region: Set[Tuple[int, int]]) -> bool:
        if len(region) not in [5, 7]:
            return False
        center = next(iter(region))  # Get any point from the set
        arms = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        arm_lengths = [sum(1 for i in range(1, 4) if (center[0] + dx * i, center[1] + dy * i) in region) for dx, dy in arms]
        return all(length > 0 for length in arm_lengths) and len(set(arm_lengths)) == 1

    def apply_transformations(grid: List[List[int]], regions_to_transform: List[Set[Tuple[int, int]]], target_color: int) -> ColoredGrid:
        new_grid = [row[:] for row in grid]
        for region in regions_to_transform:
            for x, y in region:
                new_grid[x][y] = target_color
        return ColoredGrid(values=new_grid)

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

    return apply_transformations(input_grid.values, regions_to_transform, target_color)
