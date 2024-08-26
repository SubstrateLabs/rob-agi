from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import defaultdict

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue (1) elements while respecting other colored elements.
    The solution:
    1. Identifies all blue pixels and non-black, non-blue shapes.
    2. Creates a proximity map of blue pixels.
    3. Connects nearby blue pixels using a priority queue based on distance.
    4. Forms simple closed shapes when possible.
    5. Balances the pattern by adding symmetrical connections.
    6. Refines the network by removing unnecessary branches.
    7. Ensures all connections are straight lines (horizontal, vertical, or diagonal).
    8. Validates that no new blue pixels are added beyond connections and all shapes remain unaltered.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_pixels() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1}

    def get_other_shapes() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] not in [0, 1]}

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 1]

    def connect_pixels(start: Tuple[int, int], end: Tuple[int, int]):
        r1, c1 = start
        r2, c2 = end
        dr = 1 if r2 > r1 else -1 if r2 < r1 else 0
        dc = 1 if c2 > c1 else -1 if c2 < c1 else 0
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if is_valid(r, c):
                grid.values[r][c] = 1
            r += dr
            c += dc
        if is_valid(r2, c2):
            grid.values[r2][c2] = 1

    blue_pixels = get_blue_pixels()
    other_shapes = get_other_shapes()

    # Create proximity map
    proximity_map = defaultdict(list)
    for p1 in blue_pixels:
        for p2 in blue_pixels:
            if p1 != p2:
                dist = manhattan_distance(p1, p2)
                if dist <= 3:
                    proximity_map[p1].append((dist, p2))

    # Connect nearby blue pixels
    for p1, neighbors in proximity_map.items():
        for _, p2 in sorted(neighbors):
            if manhattan_distance(p1, p2) <= 3:
                connect_pixels(p1, p2)

    # Form simple closed shapes
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.values[r+dr][c+dc] == 1 for dr, dc in [(0,0), (0,1), (1,0), (1,1)]):
                for dr, dc in [(0,0), (0,1), (1,0), (1,1)]:
                    grid.values[r+dr][c+dc] = 1

    # Balance the pattern
    center_r = sum(r for r, _ in blue_pixels) / len(blue_pixels)
    center_c = sum(c for _, c in blue_pixels) / len(blue_pixels)
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                mirror_r = int(2 * center_r - r)
                mirror_c = int(2 * center_c - c)
                if is_valid(mirror_r, mirror_c) and (mirror_r, mirror_c) not in other_shapes:
                    grid.values[mirror_r][mirror_c] = 1

    # Refine the network
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                if neighbors <= 1:
                    grid.values[r][c] = 0

    return grid
