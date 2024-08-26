from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_12eac192(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green (3) path that connects blue (1) cells
    and optionally incorporates gray (5) and sky (8) cells. The path aims to be as long
    and spread out as possible without modifying orange (7) cells.

    1. Finds potential start points (blue, gray, or sky cells).
    2. Performs DFS from each start point to find the longest path.
    3. Selects the best path based on length and spread.
    4. Extends the path to include isolated blue cells if possible.
    5. Optimizes the path by incorporating nearby gray or sky cells.
    6. Ensures the final green path is connected and orange cells are unchanged.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with the green path.
    """
    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid_cell(r+dr, c+dc)]

    def dfs(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        stack = [start]
        path = []
        visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            path.append((r, c))

            neighbors = get_neighbors(r, c)
            for nr, nc in sorted(neighbors, key=lambda x: priority_order.index(grid[x[0]][x[1]]) if grid[x[0]][x[1]] in priority_order else len(priority_order)):
                if (nr, nc) not in visited and grid[nr][nc] != 7:
                    stack.append((nr, nc))

        return path

    def path_spread(path: List[Tuple[int, int]]) -> int:
        if not path:
            return 0
        rs, cs = zip(*path)
        return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    priority_order = [1, 5, 8, 0]  # Blue, Gray, Sky, Black

    start_points = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] in [1, 5, 8]]
    if not start_points:
        return grid

    best_path = max((dfs(start) for start in start_points), key=lambda p: (len(p), path_spread(p)))

    for r, c in best_path:
        grid.set_cell(r, c, 3)

    # Extend path to include isolated blue cells
    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    for br, bc in blue_cells:
        queue = deque([(br, bc, [])])
        visited = set()
        while queue:
            r, c, path = queue.popleft()
            if grid[r][c] == 3:
                for pr, pc in path:
                    grid.set_cell(pr, pc, 3)
                break
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for nr, nc in get_neighbors(r, c):
                if grid[nr][nc] != 7 and (nr, nc) not in visited:
                    queue.append((nr, nc, path + [(r, c)]))

    # Optimize path
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in [5, 8] and any(grid[nr][nc] == 3 for nr, nc in get_neighbors(r, c)):
                grid.set_cell(r, c, 3)

    return grid
