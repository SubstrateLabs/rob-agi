from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-4 pixels are changed to the target color (red or green).
    2. Blue regions of exactly 5 pixels are changed to the target color only if they form a plus shape.
    3. The target color (red or green) is determined by the global prevalence of these colors.
    4. Larger blue regions, single blue pixels, and other colors remain unchanged.
    5. All transformations are applied simultaneously.
    6. Gray (5) acts as a border and is not considered part of any region.
    7. Only orthogonally adjacent cells are considered part of the same region.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5

    def count_colors():
        red_count = sum(row.count(RED) for row in input_grid.values)
        green_count = sum(row.count(GREEN) for row in input_grid.values)
        return red_count, green_count

    def flood_fill(x: int, y: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) not in region and 0 <= cx < rows and 0 <= cy < cols and input_grid.values[cx][cy] == BLUE:
                region.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < rows and 0 <= ny < cols and input_grid.values[nx][ny] == BLUE:
                        queue.append((nx, ny))
        return region

    def is_plus_shape(region: Set[Tuple[int, int]]) -> bool:
        if len(region) != 5:
            return False
        center = next(iter(region))
        arms = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return all((center[0] + dx, center[1] + dy) in region for dx, dy in arms)

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    red_count, green_count = count_colors()
    target_color = RED if red_count >= green_count else GREEN

    new_grid = [row[:] for row in input_grid.values]
    visited = set()
    to_transform = set()

    for i in range(rows):
        for j in range(cols):
            if input_grid.values[i][j] == BLUE and (i, j) not in visited:
                region = flood_fill(i, j)
                visited.update(region)
                size = len(region)
                if 2 <= size <= 4 or (size == 5 and is_plus_shape(region)):
                    to_transform.update(region)

    for x, y in to_transform:
        new_grid[x][y] = target_color

    return ColoredGrid(values=new_grid)
