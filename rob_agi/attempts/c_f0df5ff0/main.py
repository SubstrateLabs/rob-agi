from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a blue (1) path that connects large black (0) regions
    while avoiding other colored squares. The path is mostly thin with occasional loops and
    minimal branching. It connects significant black areas, touches grid edges when appropriate,
    and maintains the original structure of colored regions.

    1. Identify large black regions and use them as start/end points.
    2. Use A* pathfinding to connect these regions with a blue path.
    3. Add minimal branches to connect isolated black cells.
    4. Optimize the path to ensure it's mostly one cell thick.
    5. Make final adjustments for edge connections and isolated regions.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def find_large_black_regions():
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for nr, nc in get_neighbors(curr_r, curr_c):
                                if output_grid.get_cell(nr, nc) == 0:
                                    stack.append((nr, nc))
                    if len(region) > 3:  # Consider regions larger than 3 cells as "large"
                        regions.append(region)
        return regions

    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(start, goal):
        def heuristic(node):
            return manhattan_distance(node, goal)

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in get_neighbors(*current):
                tentative_g_score = g_score[current] + (1 if output_grid.get_cell(*neighbor) in [0, 1] else 2)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    large_regions = find_large_black_regions()

    if len(large_regions) >= 2:
        start_region = large_regions[0]
        end_region = max(large_regions[1:], key=lambda r: manhattan_distance(r[0], start_region[0]))
        path = a_star(start_region[0], end_region[0])
        if path:
            for r, c in path:
                output_grid.set_cell(r, c, 1)

    # Connect remaining large regions
    for region in large_regions[2:]:
        closest_blue = min((r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 1,
                           key=lambda p: min(manhattan_distance(p, cell) for cell in region))
        path = a_star(region[0], closest_blue)
        if path:
            for r, c in path:
                output_grid.set_cell(r, c, 1)

    # Add minimal branches to isolated black cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0:
                closest_blue = min((br, bc) for br in range(rows) for bc in range(cols) if output_grid.get_cell(br, bc) == 1,
                                   key=lambda p: manhattan_distance(p, (r, c)))
                if manhattan_distance((r, c), closest_blue) <= 3:  # Only connect very close black cells
                    path = a_star((r, c), closest_blue)
                    if path:
                        for pr, pc in path:
                            output_grid.set_cell(pr, pc, 1)

    # Optimize path thickness
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1:
                blue_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if output_grid.get_cell(nr, nc) == 1)
                if blue_neighbors > 2:
                    non_blue_neighbors = [n for n in get_neighbors(r, c) if output_grid.get_cell(*n) != 1]
                    if non_blue_neighbors:
                        output_grid.set_cell(r, c, output_grid.get_cell(*non_blue_neighbors[0]))

    return output_grid
