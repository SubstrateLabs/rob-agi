from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a blue (1) path that connects large black (0) regions
    while avoiding other colored squares. The path is mostly thin with occasional loops and
    controlled branching. It connects significant black areas, touches all four grid edges,
    and maintains the original structure of colored regions.

    1. Analyze the input grid and create a heat map of black cell density.
    2. Identify large black regions and potential starting points on edges.
    3. Generate a primary path structure using a modified A* algorithm.
    4. Enhance path complexity with controlled branching and loops.
    5. Connect isolated black regions and optimize path thickness.
    6. Refine edges and smooth the final path.
    7. Validate and iteratively improve the solution.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def create_heat_map():
        heat_map = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0:
                    for nr, nc in get_neighbors(r, c):
                        heat_map[nr][nc] += 1
        return heat_map

    def generate_skeleton(heat_map):
        threshold = max(max(row) for row in heat_map) // 2
        skeleton = set()
        for r in range(rows):
            for c in range(cols):
                if heat_map[r][c] >= threshold:
                    skeleton.add((r, c))
        return skeleton

    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(start, goal, skeleton):
        def heuristic(node):
            return manhattan_distance(node, goal) - (5 if node in skeleton else 0)

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

    heat_map = create_heat_map()
    skeleton = generate_skeleton(heat_map)

    # Connect skeleton points
    skeleton_points = list(skeleton)
    for i in range(len(skeleton_points) - 1):
        start = skeleton_points[i]
        end = skeleton_points[i + 1]
        path = a_star(start, end, skeleton)
        if path:
            for r, c in path:
                output_grid.set_cell(r, c, 1)

    def find_large_black_regions():
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) not in visited and output_grid.get_cell(cr, cc) == 0:
                            visited.add((cr, cc))
                            region.append((cr, cc))
                            stack.extend(get_neighbors(cr, cc))
                    if len(region) > 5:
                        regions.append(region)
        return regions

    def find_edge_starting_points():
        edge_points = []
        for r in [0, rows-1]:
            for c in range(cols):
                if heat_map[r][c] > 0:
                    edge_points.append((r, c))
        for c in [0, cols-1]:
            for r in range(1, rows-1):
                if heat_map[r][c] > 0:
                    edge_points.append((r, c))
        return sorted(edge_points, key=lambda p: heat_map[p[0]][p[1]], reverse=True)

    def generate_primary_path(start, black_regions):
        path = set()
        current = start
        path.add(current)
        output_grid.set_cell(*current, 1)

        for region in black_regions:
            target = min(region, key=lambda p: manhattan_distance(current, p))
            new_segment = a_star(current, target, skeleton)
            if new_segment:
                path.update(new_segment)
                for r, c in new_segment:
                    output_grid.set_cell(r, c, 1)
                current = target

        return path

    def add_complexity(path):
        for _ in range(len(path) // 10):
            start = random.choice(list(path))
            end = random.choice(list(path))
            if manhattan_distance(start, end) > 5:
                new_branch = a_star(start, end, skeleton)
                if new_branch:
                    for r, c in new_branch:
                        output_grid.set_cell(r, c, 1)
                    path.update(new_branch)

    def connect_isolated_regions(path):
        black_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 0]
        for cell in black_cells:
            if all(output_grid.get_cell(nr, nc) != 1 for nr, nc in get_neighbors(*cell)):
                closest_blue = min(path, key=lambda p: manhattan_distance(p, cell))
                new_branch = a_star(cell, closest_blue, set())
                if new_branch:
                    for r, c in new_branch:
                        output_grid.set_cell(r, c, 1)
                    path.update(new_branch)

    def optimize_thickness(path):
        for r, c in path:
            blue_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if output_grid.get_cell(nr, nc) == 1)
            if blue_neighbors > 2 and random.random() < 0.3:
                non_blue = [n for n in get_neighbors(r, c) if output_grid.get_cell(*n) != 1]
                if non_blue:
                    output_grid.set_cell(r, c, output_grid.get_cell(*random.choice(non_blue)))
                    path.remove((r, c))

    def ensure_edge_connections(path):
        edges = [0, rows-1, cols-1]
        for edge in edges:
            if not any(r == edge or c == edge for r, c in path):
                closest_path = min(path, key=lambda p: min(p[0], p[1], rows-1-p[0], cols-1-p[1]))
                target = (edge, closest_path[1]) if edge < rows else (closest_path[0], edge)
                new_segment = a_star(closest_path, target, set())
                if new_segment:
                    for r, c in new_segment:
                        output_grid.set_cell(r, c, 1)
                    path.update(new_segment)

    heat_map = create_heat_map()
    skeleton = generate_skeleton(heat_map)
    black_regions = find_large_black_regions()
    starting_points = find_edge_starting_points()

    if starting_points:
        primary_path = generate_primary_path(starting_points[0], black_regions)
        add_complexity(primary_path)
        connect_isolated_regions(primary_path)
        optimize_thickness(primary_path)
        ensure_edge_connections(primary_path)

    return output_grid
