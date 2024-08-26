from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_12eac192(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green (3) path that connects blue (1) cells,
    incorporating gray (5) and sky (8) cells when convenient. The path aims to be simple
    and continuous, while preserving much of the original grid structure.

    1. Identifies blue cells and groups them into clusters.
    2. For each cluster, creates a green path using BFS, prioritizing connections to other blue cells.
    3. Incorporates gray and sky cells when they help connect blue cells without complicating the path.
    4. Leaves some blue cells unconverted if connecting them would create an overly complex path.
    5. Ensures the final green path is continuous and orange (7) cells remain unchanged.
    6. Maintains the overall structure of the original grid.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with the green path.
    """
    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid_cell(r+dr, c+dc)]

    def bfs(start: Tuple[int, int], cluster: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        queue = deque([(start, 0)])
        path = []
        visited = set()
        non_blue_count = 0

        while queue:
            (r, c), non_blue = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            path.append((r, c))

            if grid[r][c] != 1:
                non_blue_count += 1
            else:
                non_blue_count = 0

            if non_blue_count > 3:
                break

            neighbors = get_neighbors(r, c)
            for nr, nc in sorted(neighbors, key=lambda x: priority_order.index(grid[x[0]][x[1]]) if grid[x[0]][x[1]] in priority_order else len(priority_order)):
                if (nr, nc) not in visited and grid[nr][nc] != 7:
                    new_non_blue = non_blue + 1 if grid[nr][nc] != 1 else 0
                    queue.append(((nr, nc), new_non_blue))

            if all((r, c) in visited for r, c in cluster):
                break

        return path

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    priority_order = [1, 5, 8, 0]  # Blue, Gray, Sky, Black

    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not blue_cells:
        return grid

    # Group blue cells into clusters
    clusters = []
    visited_blues = set()
    for blue in blue_cells:
        if blue not in visited_blues:
            cluster = set()
            stack = [blue]
            while stack:
                cell = stack.pop()
                if cell not in visited_blues:
                    visited_blues.add(cell)
                    cluster.add(cell)
                    stack.extend([n for n in get_neighbors(*cell) if grid[n[0]][n[1]] == 1 and n not in visited_blues])
            clusters.append(cluster)

    # Process each cluster
    for cluster in sorted(clusters, key=len, reverse=True):
        start = max(cluster, key=lambda x: sum(1 for nr, nc in get_neighbors(*x) if grid[nr][nc] in [1, 5, 8]))
        path = bfs(start, cluster)

        # Convert path to green
        for r, c in path:
            if grid[r][c] in [0, 1, 5, 8]:
                grid.set_cell(r, c, 3)

    # Try to connect remaining blue cells
    for br, bc in blue_cells:
        if grid[br][bc] == 1:
            nearest_green = min((abs(br-r) + abs(bc-c), (r, c)) for r in range(rows) for c in range(cols) if grid[r][c] == 3)[1]
            current_path = []
            r, c = br, bc
            while (r, c) != nearest_green:
                if grid[r][c] in [0, 1, 5, 8]:
                    current_path.append((r, c))
                dr = 1 if r < nearest_green[0] else -1 if r > nearest_green[0] else 0
                dc = 1 if c < nearest_green[1] else -1 if c > nearest_green[1] else 0
                r, c = r + dr, c + dc
            if len(current_path) <= 3:  # Only connect if path is short
                for pr, pc in current_path:
                    grid.set_cell(pr, pc, 3)

    return grid
