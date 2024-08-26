from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_54db823b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving colored regions that are:
    1. Connected to the edge of the grid.
    2. Reachable through a path of adjacent colored cells from an edge-connected region.
    All other isolated regions are removed (set to black).

    The algorithm works in three main steps:
    1. Identifies all edge-connected regions.
    2. Propagates reachability from edge-connected regions.
    3. Applies changes, keeping reachable regions and removing isolated ones.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    reachable = [[False for _ in range(cols)] for _ in range(rows)]

    def is_edge(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def dfs_edge(r: int, c: int, color: int):
        if not (0 <= r < rows and 0 <= c < cols) or visited[r][c] or input_grid.values[r][c] != color:
            return
        visited[r][c] = True
        reachable[r][c] = True
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs_edge(r + dr, c + dc, color)

    def propagate_reachability():
        queue = [(r, c) for r in range(rows) for c in range(cols) if reachable[r][c]]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid.values[nr][nc] != 0:
                    visited[nr][nc] = True
                    reachable[nr][nc] = True
                    queue.append((nr, nc))

    # Step 1: Identify edge-connected regions
    for r in range(rows):
        for c in range(cols):
            if is_edge(r, c) and not visited[r][c] and input_grid.values[r][c] != 0:
                dfs_edge(r, c, input_grid.values[r][c])

    # Step 2: Propagate reachability
    propagate_reachability()

    # Step 3: Apply changes
    for r in range(rows):
        for c in range(cols):
            if not reachable[r][c] and output_grid.values[r][c] != 0:
                output_grid.values[r][c] = 0

    return output_grid
