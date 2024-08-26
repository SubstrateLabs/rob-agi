from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a blue (1) path that connects large black (0) regions
    while avoiding other colored squares. The path is mostly thin with occasional loops and
    minimal branching. It connects significant black areas, touches grid edges when appropriate,
    and maintains the original structure of colored regions.

    1. Analyze the input grid and create a heat map of black cell density.
    2. Generate a skeleton structure connecting high-density black areas.
    3. Refine the path using a modified A* algorithm.
    4. Add complexity and balance with branches and loops.
    5. Optimize path thickness and create deliberate loops.
    6. Connect isolated black cells and make final refinements.
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

    # Add complexity and balance
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0 and heat_map[r][c] > 0:
                closest_blue = min((br, bc) for br in range(rows) for bc in range(cols) 
                                   if output_grid.get_cell(br, bc) == 1,
                                   key=lambda p: manhattan_distance(p, (r, c)))
                if manhattan_distance((r, c), closest_blue) <= 5:
                    path = a_star((r, c), closest_blue, skeleton)
                    if path:
                        for pr, pc in path:
                            output_grid.set_cell(pr, pc, 1)

    # Optimize path thickness and create loops
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1:
                blue_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if output_grid.get_cell(nr, nc) == 1)
                if blue_neighbors > 2:
                    non_blue_neighbors = [n for n in get_neighbors(r, c) if output_grid.get_cell(*n) != 1]
                    if non_blue_neighbors and heat_map[r][c] < 2:
                        output_grid.set_cell(r, c, output_grid.get_cell(*non_blue_neighbors[0]))
                elif blue_neighbors == 1 and heat_map[r][c] > 1:
                    for nr, nc in get_neighbors(r, c):
                        if output_grid.get_cell(nr, nc) == 0:
                            output_grid.set_cell(nr, nc, 1)
                            break

    return output_grid
