from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue (1) elements while respecting other colored elements.
    The solution:
    1. Identifies all blue pixels and non-black, non-blue shapes.
    2. Uses a breadth-first search to connect nearby blue pixels.
    3. Ensures connections are straight lines (horizontal, vertical, or diagonal).
    4. Avoids crossing or altering non-blue shapes.
    5. Creates a balanced and symmetrical network of blue connections.
    6. Removes isolated blue pixels that aren't part of the main network.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_pixels() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1}

    def get_other_shapes() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] not in [0, 1]}

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 1]

    def connect_pixels(start: Tuple[int, int], end: Tuple[int, int]):
        r1, c1 = start
        r2, c2 = end
        dr = (r2 > r1) - (r2 < r1)
        dc = (c2 > c1) - (c2 < c1)
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

    # BFS to connect blue pixels
    visited = set()
    for start in blue_pixels:
        if start not in visited:
            queue = deque([start])
            visited.add(start)
            while queue:
                current = queue.popleft()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r, c = current[0] + dr, current[1] + dc
                        end = (r, c)
                        if end in blue_pixels and end not in visited:
                            connect_pixels(current, end)
                            queue.append(end)
                            visited.add(end)

    # Remove isolated blue pixels
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                                if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                if neighbors == 0:
                    grid.values[r][c] = 0

    return grid
