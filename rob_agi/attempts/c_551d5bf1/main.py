from rob_agi.colored_grid import ColoredGrid
from collections import deque
import heapq

def solve_551d5bf1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a sky blue (8) network that connects
    all blue (1) structures, extends to the right and bottom edges, and preserves
    the original blue structures.

    The function identifies blue structures, creates a minimal spanning tree to
    connect them, extends the network to the edges, fills the interiors of blue
    structures, and optimizes the network to use the minimum necessary sky blue cells.
    It avoids diagonal connections and ensures the network forms a tree-like structure
    without cycles.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def find_blue_structures():
        structures = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 1 and (r, c) not in visited:
                    structure = []
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == 1:
                            visited.add((curr_r, curr_c))
                            structure.append((curr_r, curr_c))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if is_valid(nr, nc):
                                    queue.append((nr, nc))
                    structures.append(structure)
        return structures

    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def create_graph(structures):
        graph = []
        for i, s1 in enumerate(structures):
            for j, s2 in enumerate(structures[i+1:], i+1):
                dist = min(manhattan_distance(p1, p2) for p1 in s1 for p2 in s2)
                graph.append((dist, i, j))
        return graph

    def find_mst(graph, n):
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True

        mst = []
        graph.sort()
        for w, u, v in graph:
            if union(u, v):
                mst.append((u, v))
            if len(mst) == n - 1:
                break
        return mst

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

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if not is_valid(*neighbor) or output_grid.values[neighbor[0]][neighbor[1]] == 1:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    # Find blue structures
    structures = find_blue_structures()

    # Create graph and find MST
    graph = create_graph(structures)
    mst = find_mst(graph, len(structures))

    # Create sky blue network
    for u, v in mst:
        start = structures[u][0]
        end = structures[v][0]
        path = a_star(start, end)
        if path:
            for r, c in path:
                if output_grid.values[r][c] != 1:
                    output_grid.values[r][c] = 8

    # Extend to right edge and bottom row
    for r in range(rows):
        rightmost = max(c for c in range(cols) if output_grid.values[r][c] in [1, 8])
        for c in range(rightmost + 1, cols):
            output_grid.values[r][c] = 8

    for c in range(cols):
        bottommost = max(r for r in range(rows) if output_grid.values[r][c] in [1, 8])
        for r in range(bottommost + 1, rows):
            output_grid.values[r][c] = 8

    # Fill blue structure interiors
    for structure in structures:
        for r, c in structure:
            if r > 0 and c > 0 and r < rows - 1 and c < cols - 1:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = 8

    # Optimize sky blue network
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 8:
                neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if is_valid(r+dr, c+dc) and output_grid.values[r+dr][c+dc] in [1, 8])
                if neighbors <= 1 and r < rows - 1 and c < cols - 1:
                    output_grid.values[r][c] = 0

    return output_grid
