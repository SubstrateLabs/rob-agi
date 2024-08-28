from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque
import heapq

def solve_f3b10344(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f3b10344 challenge by creating a sky blue network that connects non-black shapes.
    
    The function analyzes the grid, identifies non-black shapes, creates a graph representation,
    generates a minimum spanning tree to find optimal connections, and then creates a 3-cell wide
    sky blue network connecting all shapes. It handles edge cases, optimizes the network,
    and ensures all original shapes are preserved.
    """
    BLACK, SKY_BLUE = 0, 8
    
    if all(cell == BLACK for row in input_grid.values for cell in row):
        return input_grid

    rows, cols = input_grid.get_dimensions()
    grid = input_grid.deep_copy()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(r+dr, c+dc)]

    def find_shapes() -> Dict[int, Set[Tuple[int, int]]]:
        shapes = {}
        visited = set()
        shape_id = 0
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != BLACK and (r, c) not in visited:
                    shape = set()
                    color = input_grid.values[r][c]
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited:
                            visited.add((curr_r, curr_c))
                            shape.add((curr_r, curr_c))
                            for nr, nc in get_neighbors(curr_r, curr_c):
                                if input_grid.values[nr][nc] == color:
                                    queue.append((nr, nc))
                    shapes[shape_id] = shape
                    shape_id += 1
        return shapes

    def calculate_distances(shapes: Dict[int, Set[Tuple[int, int]]]) -> Dict[Tuple[int, int], int]:
        distances = {}
        for id1, shape1 in shapes.items():
            for id2, shape2 in shapes.items():
                if id1 < id2:
                    dist = min(abs(r1-r2) + abs(c1-c2) for r1, c1 in shape1 for r2, c2 in shape2)
                    distances[(id1, id2)] = dist
        return distances

    def prim_mst(shapes: Dict[int, Set[Tuple[int, int]]], distances: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
        mst = []
        start = next(iter(shapes))
        visited = {start}
        edges = [(dist, start, end) for (s, end), dist in distances.items() if s == start]
        heapq.heapify(edges)

        while edges and len(visited) < len(shapes):
            dist, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                mst.append((u, v))
                for end in shapes:
                    if end not in visited:
                        if (v, end) in distances:
                            heapq.heappush(edges, (distances[(v, end)], v, end))
                        elif (end, v) in distances:
                            heapq.heappush(edges, (distances[(end, v)], v, end))

        return mst

    def create_sky_blue_path(start: Tuple[int, int], end: Tuple[int, int]):
        queue = [(0, start, [])]
        visited = set()
        while queue:
            cost, current, path = heapq.heappop(queue)
            if current == end:
                for r, c in path:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == BLACK:
                                grid.values[r+dr][c+dc] = SKY_BLUE
                return
            if current not in visited:
                visited.add(current)
                for nr, nc in get_neighbors(*current):
                    if (nr, nc) not in visited:
                        new_cost = cost + 1
                        heapq.heappush(queue, (new_cost, (nr, nc), path + [(nr, nc)]))

    shapes = find_shapes()
    if len(shapes) == 1:
        shape = next(iter(shapes.values()))
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        for r in range(max(0, min_r-3), min(rows, max_r+4)):
            for c in range(max(0, min_c-3), min(cols, max_c+4)):
                if (r, c) not in shape:
                    grid.values[r][c] = SKY_BLUE
    else:
        distances = calculate_distances(shapes)
        mst = prim_mst(shapes, distances)
        for u, v in mst:
            start = next(iter(shapes[u]))
            end = next(iter(shapes[v]))
            create_sky_blue_path(start, end)

    # Post-processing: fill small gaps and create border if needed
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == BLACK:
                sky_blue_neighbors = sum(1 for nr, nc in get_neighbors(r, c) if grid.values[nr][nc] == SKY_BLUE)
                if sky_blue_neighbors >= 2:
                    grid.values[r][c] = SKY_BLUE
            if r < 3 or r >= rows - 3 or c < 3 or c >= cols - 3:
                if grid.values[r][c] == BLACK:
                    grid.values[r][c] = SKY_BLUE

    # Restore original shapes
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != BLACK:
                grid.values[r][c] = input_grid.values[r][c]

    return grid
