from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e5790162(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green path that connects the initial green square
    to all magenta and sky blue squares. The path prioritizes horizontal movement, then vertical.
    It doesn't overwrite magenta or sky blue squares and can continue past them if needed.
    
    1. Finds the starting green square.
    2. Identifies all magenta and sky blue target squares.
    3. Creates a path to the nearest target, prioritizing horizontal movement.
    4. Continues the path to other targets if they're in the same direction.
    5. Repeats until all targets are connected.
    """
    def find_start() -> Tuple[int, int]:
        for r, row in enumerate(input_grid.values):
            for c, val in enumerate(row):
                if val == 3:
                    return r, c
        return -1, -1  # Should never happen if input is valid

    def find_targets() -> List[Tuple[int, int]]:
        return [(r, c) for r, row in enumerate(input_grid.values) 
                for c, val in enumerate(row) if val in (6, 8)]

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_nearest_target(pos: Tuple[int, int], targets: List[Tuple[int, int]]) -> Tuple[int, int]:
        return min(targets, key=lambda t: (manhattan_distance(pos, t), abs(t[1] - pos[1])))

    def draw_path(start: Tuple[int, int], end: Tuple[int, int], grid: List[List[int]]):
        r, c = start
        while (r, c) != end:
            if c != end[1]:
                c += 1 if c < end[1] else -1
            elif r != end[0]:
                r += 1 if r < end[0] else -1
            if grid[r][c] not in (6, 8):
                grid[r][c] = 3
            if (r, c) == end:
                break

    output_grid = [row[:] for row in input_grid.values]
    start = find_start()
    targets = find_targets()
    current = start

    while targets:
        nearest = find_nearest_target(current, targets)
        draw_path(current, nearest, output_grid)
        current = nearest
        targets.remove(nearest)

    return ColoredGrid(values=output_grid)
