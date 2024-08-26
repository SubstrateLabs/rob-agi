from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a thin blue (1) path that follows color boundaries,
    connects large black (0) regions, and touches all four edges of the grid. The path
    preserves the original structure and color patterns as much as possible.

    1. Analyze the input grid to identify color boundaries and edges of black regions.
    2. Create an initial path following these boundaries, starting from a corner.
    3. Ensure the path touches all four edges of the grid.
    4. Refine the path to maintain thinness and follow significant color boundaries.
    5. Make minimal adjustments to break up large mono-color regions if necessary.
    6. Optimize the path to effectively separate different color regions.
    7. Validate and iteratively refine the solution.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_boundary(r, c):
        color = output_grid.get_cell(r, c)
        return any(output_grid.get_cell(nr, nc) != color for nr, nc in get_neighbors(r, c))

    def find_start_point():
        for r in range(rows):
            for c in range(cols):
                if is_boundary(r, c):
                    return (r, c)
        return (0, 0)  # Fallback to top-left corner

    def dfs_path(start):
        stack = [start]
        path = set()
        visited = set()

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                r, c = current
                if is_boundary(r, c):
                    path.add(current)
                    output_grid.set_cell(r, c, 1)
                    neighbors = [n for n in get_neighbors(r, c) if n not in visited and is_boundary(*n)]
                    stack.extend(neighbors)

        return path

    def ensure_edge_connections(path):
        edges = [(0, c) for c in range(cols)] + [(rows-1, c) for c in range(cols)] + \
                [(r, 0) for r in range(rows)] + [(r, cols-1) for r in range(rows)]
        
        for edge in edges:
            if edge not in path:
                nearest = min(path, key=lambda p: abs(p[0]-edge[0]) + abs(p[1]-edge[1]))
                current = nearest
                while current != edge:
                    r, c = current
                    next_step = min(get_neighbors(r, c), key=lambda n: abs(n[0]-edge[0]) + abs(n[1]-edge[1]))
                    path.add(next_step)
                    output_grid.set_cell(*next_step, 1)
                    current = next_step

    def optimize_path(path):
        for r, c in path:
            neighbors = get_neighbors(r, c)
            blue_neighbors = sum(1 for nr, nc in neighbors if output_grid.get_cell(nr, nc) == 1)
            if blue_neighbors > 2:
                non_blue = [n for n in neighbors if output_grid.get_cell(*n) != 1]
                if non_blue:
                    output_grid.set_cell(r, c, output_grid.get_cell(*non_blue[0]))
                    path.remove((r, c))

    start = find_start_point()
    path = dfs_path(start)
    ensure_edge_connections(path)
    optimize_path(path)

    return output_grid
