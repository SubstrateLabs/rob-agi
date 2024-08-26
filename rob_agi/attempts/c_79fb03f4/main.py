from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Connecting all initial blue (1) points with straight lines
    2. Extending blue lines to grid edges where possible
    3. Filling in enclosed areas
    4. Respecting other colored squares as barriers

    The function uses BFS to find shortest paths between blue points,
    extends lines to edges, and fills in enclosed areas.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid_move(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) in [0, 1]

    def find_shortest_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == end:
                return path
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid_move(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def connect_points(points: List[Tuple[int, int]]):
        for i in range(1, len(points)):
            path = find_shortest_path(points[i-1], points[i])
            for r, c in path:
                grid.set_cell(r, c, 1)

    def extend_to_edges():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 1:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        while is_valid_move(nr, nc):
                            grid.set_cell(nr, nc, 1)
                            nr, nc = nr + dr, nc + dc

    def fill_enclosed_areas():
        changed = True
        while changed:
            changed = False
            for r in range(rows):
                for c in range(cols):
                    if grid.get_cell(r, c) == 0:
                        if all(grid.get_cell(r+dr, c+dc) == 1 for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)] if 0 <= r+dr < rows and 0 <= c+dc < cols):
                            grid.set_cell(r, c, 1)
                            changed = True

    # Find all initial blue points
    blue_points = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 1]

    # Connect blue points
    connect_points(blue_points)

    # Extend blue lines to edges
    extend_to_edges()

    # Fill enclosed areas
    fill_enclosed_areas()

    return grid
