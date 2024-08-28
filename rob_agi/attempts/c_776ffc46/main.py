from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
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

    The solution uses a flood fill algorithm to identify connected blue regions,
    determines the target color based on the prevalence of red and green,
    and applies transformations to eligible blue regions simultaneously.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5

    def count_colors():
        red_count = sum(row.count(RED) for row in input_grid.values)
        green_count = sum(row.count(GREEN) for row in input_grid.values)
        return red_count, green_count

    def flood_fill(x: int, y: int, visited: List[List[bool]]) -> List[Tuple[int, int]]:
        region = []
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if (0 <= cx < rows and 0 <= cy < cols and
                input_grid.values[cx][cy] == BLUE and not visited[cx][cy]):
                visited[cx][cy] = True
                region.append((cx, cy))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < rows and 0 <= ny < cols and input_grid.values[nx][ny] != GRAY:
                        queue.append((nx, ny))
        return region

    def is_plus_shape(region: List[Tuple[int, int]]) -> bool:
        if len(region) != 5:
            return False
        center = min(region, key=lambda p: abs(p[0] - rows // 2) + abs(p[1] - cols // 2))
        orthogonal_neighbors = [(center[0]+1, center[1]), (center[0]-1, center[1]),
                                (center[0], center[1]+1), (center[0], center[1]-1)]
        return all(neighbor in region for neighbor in orthogonal_neighbors)

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    red_count, green_count = count_colors()
    target_color = RED if red_count >= green_count else GREEN

    visited = [[False for _ in row] for row in input_grid.values]
    transform_markers = [[False for _ in row] for row in input_grid.values]

    for i in range(rows):
        for j in range(cols):
            if input_grid.values[i][j] == BLUE and not visited[i][j]:
                region = flood_fill(i, j, visited)
                if 2 <= len(region) <= 4 or (len(region) == 5 and is_plus_shape(region)):
                    for x, y in region:
                        transform_markers[x][y] = True

    transformed_grid = [row[:] for row in input_grid.values]
    for i in range(rows):
        for j in range(cols):
            if transform_markers[i][j]:
                transformed_grid[i][j] = target_color

    return ColoredGrid(values=transformed_grid)
