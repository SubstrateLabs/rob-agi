from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque
import heapq

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal red structure that connects all blue squares.
    
    1. Analyze the input grid to identify blue segments and existing red squares.
    2. Create a graph representation of blue segments and existing red squares.
    3. Find the Minimum Spanning Tree (MST) to connect all elements.
    4. Convert the MST to a grid structure, adding red squares as needed.
    5. Optimize the red structure by removing unnecessary red squares.
    6. Validate the final structure to ensure all blue squares and existing red squares are connected.
    
    Returns a new grid with the minimal red structure added while preserving all blue squares and existing red squares.
    """
    output_grid = input_grid.deep_copy()
    elements = find_elements(input_grid)
    graph = create_graph(elements)
    mst = minimum_spanning_tree(graph, len(elements[2]) + len(elements[8]))
    add_red_structure(output_grid, mst, elements)
    connect_existing_red(output_grid, input_grid)
    optimize_red_structure(output_grid)
    return output_grid

def find_elements(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int, int, int]]]:
    rows, cols = grid.get_dimensions()
    elements = {2: [], 8: []}
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] in {2, 8} and (r, c) not in visited:
                segment = get_segment(grid, r, c, visited)
                elements[grid.values[r][c]].append(segment)
    
    return elements

def get_segment(grid: ColoredGrid, start_r: int, start_c: int, visited: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    queue = deque([(start_r, start_c)])
    min_r, min_c, max_r, max_c = start_r, start_c, start_r, start_c
    color = grid.values[start_r][start_c]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) not in visited and grid.values[r][c] == color:
            visited.add((r, c))
            min_r, min_c = min(min_r, r), min(min_c, c)
            max_r, max_c = max(max_r, r), max(max_c, c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))
    
    return (min_r, min_c, max_r, max_c)

def create_graph(elements: Dict[int, List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int]]:
    graph = []
    all_elements = elements[2] + elements[8]
    for i, (r1, c1, r2, c2) in enumerate(all_elements):
        for j, (r3, c3, r4, c4) in enumerate(all_elements[i+1:], i+1):
            dist = abs(r2 - r3) + abs(c2 - c3)
            graph.append((dist, i, j))
    return graph

def minimum_spanning_tree(graph: List[Tuple[int, int, int]], num_elements: int) -> List[Tuple[int, int, int]]:
    graph.sort()
    parent = list(range(num_elements))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    mst = []
    for w, u, v in graph:
        if find(u) != find(v):
            union(u, v)
            mst.append((w, u, v))
    
    return mst

def add_red_structure(grid: ColoredGrid, mst: List[Tuple[int, int, int]], elements: Dict[int, List[Tuple[int, int, int, int]]]):
    all_elements = elements[2] + elements[8]
    for _, u, v in mst:
        r1, c1, r2, c2 = all_elements[u]
        r3, c3, r4, c4 = all_elements[v]
        connect_segments(grid, r1, c1, r2, c2, r3, c3, r4, c4)

def connect_segments(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int, r3: int, c3: int, r4: int, c4: int):
    start = ((r1 + r2) // 2, (c1 + c2) // 2)
    end = ((r3 + r4) // 2, (c3 + c4) // 2)
    path = find_path(grid, start, end)
    for r, c in path:
        if grid.values[r][c] == 0:
            grid.values[r][c] = 2

def find_path(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    queue = [(0, start, [])]
    visited = set()
    
    while queue:
        cost, (r, c), path = heapq.heappop(queue)
        if (r, c) == end:
            return path + [(r, c)]
        if (r, c) not in visited:
            visited.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    new_cost = cost + (1 if grid.values[nr][nc] == 0 else 0)
                    heapq.heappush(queue, (new_cost, (nr, nc), path + [(r, c)]))
    
    return []

def connect_existing_red(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2 and output_grid.values[r][c] != 2:
                nearest_red = find_nearest_red(output_grid, r, c)
                if nearest_red:
                    path = find_path(output_grid, (r, c), nearest_red)
                    for pr, pc in path:
                        if output_grid.values[pr][pc] == 0:
                            output_grid.values[pr][pc] = 2

def find_nearest_red(grid: ColoredGrid, r: int, c: int) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    queue = deque([(r, c, 0)])
    visited = set()
    
    while queue:
        cr, cc, dist = queue.popleft()
        if (cr, cc) not in visited:
            visited.add((cr, cc))
            if grid.values[cr][cc] == 2:
                return (cr, cc)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc, dist + 1))
    
    return None

def optimize_red_structure(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2:
                temp_value = grid.values[r][c]
                grid.values[r][c] = 0
                if is_connected(grid):
                    continue
                grid.values[r][c] = temp_value

def is_connected(grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    start = next((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] in {2, 8})
    queue = deque([start])
    visited = set()
    
    while queue:
        r, c = queue.popleft()
        if (r, c) not in visited:
            visited.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in {2, 8}:
                    queue.append((nr, nc))
    
    return all((r, c) in visited for r in range(rows) for c in range(cols) if grid.values[r][c] in {2, 8})
