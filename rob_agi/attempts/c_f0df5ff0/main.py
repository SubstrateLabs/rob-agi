from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a thin blue (1) path that follows color boundaries,
    connects large regions, and touches all four edges of the grid. The path preserves the
    original structure and color patterns as much as possible.

    1. Analyze the input grid to identify color boundaries and create a heat map.
    2. Generate an initial path following these boundaries, starting from a high-density corner.
    3. Ensure the path touches all four edges of the grid.
    4. Optimize the path to maintain thinness and effectively separate different color regions.
    5. Handle large monochromatic regions by adding strategic cuts.
    6. Balance path distribution across the grid.
    7. Fine-tune the solution to enclose color regions and separate remaining large areas.
    8. Validate and iteratively refine the solution.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_boundary(r, c):
        color = output_grid.get_cell(r, c)
        return any(output_grid.get_cell(nr, nc) != color for nr, nc in get_neighbors(r, c))

    def create_heat_map():
        heat_map = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if is_boundary(r, c):
                    heat_map[r][c] = 1
                    for nr, nc in get_neighbors(r, c):
                        heat_map[nr][nc] += 1
        return heat_map

    def find_start_point(heat_map):
        corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        return max(corners, key=lambda p: heat_map[p[0]][p[1]])

    def generate_path(start, heat_map):
        path = set()
        visited = set()
        queue = deque([(start, [])])

        while queue:
            current, path_so_far = queue.popleft()
            if current not in visited:
                visited.add(current)
                path_so_far = path_so_far + [current]
                r, c = current

                if is_boundary(r, c) or len(path_so_far) > 1:
                    for p in path_so_far:
                        path.add(p)
                        output_grid.set_cell(*p, 1)

                neighbors = sorted(
                    [n for n in get_neighbors(r, c) if n not in visited],
                    key=lambda n: (-heat_map[n[0]][n[1]], len(path_so_far))
                )
                for neighbor in neighbors:
                    queue.append((neighbor, []))

        return path

    def ensure_edge_connections(path):
        edges = set([(0, c) for c in range(cols)] + [(rows-1, c) for c in range(cols)] +
                    [(r, 0) for r in range(rows)] + [(r, cols-1) for r in range(rows)])
        unconnected = edges - path
        
        for edge in unconnected:
            nearest = min(path, key=lambda p: abs(p[0]-edge[0]) + abs(p[1]-edge[1]))
            current = nearest
            while current != edge:
                r, c = current
                next_step = min(get_neighbors(r, c), key=lambda n: abs(n[0]-edge[0]) + abs(n[1]-edge[1]))
                path.add(next_step)
                output_grid.set_cell(*next_step, 1)
                current = next_step

    def optimize_path(path):
        for r, c in list(path):
            neighbors = get_neighbors(r, c)
            blue_neighbors = sum(1 for nr, nc in neighbors if output_grid.get_cell(nr, nc) == 1)
            if blue_neighbors > 2:
                non_blue = [n for n in neighbors if output_grid.get_cell(*n) != 1]
                if non_blue:
                    output_grid.set_cell(r, c, output_grid.get_cell(*non_blue[0]))
                    path.remove((r, c))

    def handle_large_regions(path):
        def flood_fill(r, c, color):
            region = set()
            stack = [(r, c)]
            while stack:
                current = stack.pop()
                if current not in region and output_grid.get_cell(*current) == color:
                    region.add(current)
                    stack.extend(get_neighbors(*current))
            return region

        threshold = rows * cols // 16  # Adjust this threshold as needed
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in path:
                    region = flood_fill(r, c, output_grid.get_cell(r, c))
                    if len(region) > threshold:
                        center = (sum(x for x, _ in region) // len(region),
                                  sum(y for _, y in region) // len(region))
                        nearest_path = min(path, key=lambda p: abs(p[0]-center[0]) + abs(p[1]-center[1]))
                        current = nearest_path
                        while current != center:
                            r, c = current
                            next_step = min(get_neighbors(r, c), key=lambda n: abs(n[0]-center[0]) + abs(n[1]-center[1]))
                            if next_step not in path:
                                path.add(next_step)
                                output_grid.set_cell(*next_step, 1)
                            current = next_step

    heat_map = create_heat_map()
    start = find_start_point(heat_map)
    path = generate_path(start, heat_map)
    ensure_edge_connections(path)
    optimize_path(path)
    handle_large_regions(path)

    return output_grid
