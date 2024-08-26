from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

from typing import List, Tuple, Set
from collections import deque
import heapq

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal red structure that connects all blue squares.
    
    1. Analyze the input grid to identify blue segments and squares.
    2. Create a graph representation of blue segments and squares.
    3. Find the Minimum Spanning Tree (MST) to connect all blue elements.
    4. Convert the MST to a grid structure, adding red squares as needed.
    5. Incorporate any existing red squares from the input.
    6. Optimize the red structure by removing unnecessary red squares.
    7. Validate the final structure to ensure all blue squares are connected.
    
    Returns a new grid with the minimal red structure added while preserving all blue squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find all blue squares and segments
    blue_elements = find_blue_elements(input_grid)
    
    # Create graph representation
    graph = create_graph(blue_elements)
    
    # Find Minimum Spanning Tree
    mst = minimum_spanning_tree(graph)
    
    # Convert MST to grid structure
    add_red_structure(output_grid, mst, blue_elements)
    
    # Incorporate existing red squares
    connect_existing_red(output_grid, input_grid)
    
    # Optimize red structure
    optimize_red_structure(output_grid)
    
    return output_grid

def find_blue_elements(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    blue_elements = []
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                segment = get_blue_segment(grid, r, c, visited)
                blue_elements.append(segment)
    
    return blue_elements

def get_blue_segment(grid: ColoredGrid, start_r: int, start_c: int, visited: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    queue = deque([(start_r, start_c)])
    min_r, min_c, max_r, max_c = start_r, start_c, start_r, start_c
    
    while queue:
        r, c = queue.popleft()
        if (r, c) not in visited and grid.values[r][c] == 8:
            visited.add((r, c))
            min_r, min_c = min(min_r, r), min(min_c, c)
            max_r, max_c = max(max_r, r), max(max_c, c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))
    
    return (min_r, min_c, max_r, max_c)

def create_graph(blue_elements: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
    graph = []
    for i, (r1, c1, r2, c2) in enumerate(blue_elements):
        for j, (r3, c3, r4, c4) in enumerate(blue_elements[i+1:], i+1):
            dist = min(abs(r2 - r3), abs(r1 - r4)) + min(abs(c2 - c3), abs(c1 - c4))
            graph.append((dist, i, j))
    return graph

def minimum_spanning_tree(graph: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    graph.sort()
    parent = list(range(len(graph)))
    
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

def add_red_structure(grid: ColoredGrid, mst: List[Tuple[int, int, int]], blue_elements: List[Tuple[int, int, int, int]]):
    for _, u, v in mst:
        r1, c1, r2, c2 = blue_elements[u]
        r3, c3, r4, c4 = blue_elements[v]
        connect_segments(grid, r1, c1, r2, c2, r3, c3, r4, c4)

def connect_segments(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int, r3: int, c3: int, r4: int, c4: int):
    if r1 <= r4 and r3 <= r2:  # Vertical overlap
        c_connect = c2 if abs(c2 - c3) < abs(c1 - c4) else c1
        for r in range(min(r1, r3), max(r2, r4) + 1):
            if grid.values[r][c_connect] == 0:
                grid.values[r][c_connect] = 2
    else:  # Horizontal connection
        r_connect = r2 if abs(r2 - r3) < abs(r1 - r4) else r1
        for c in range(min(c1, c3), max(c2, c4) + 1):
            if grid.values[r_connect][c] == 0:
                grid.values[r_connect][c] = 2

def connect_existing_red(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2 and output_grid.values[r][c] == 0:
                connect_to_nearest_red(output_grid, r, c)

def connect_to_nearest_red(grid: ColoredGrid, start_r: int, start_c: int):
    rows, cols = grid.get_dimensions()
    queue = [(0, start_r, start_c)]
    visited = set()
    
    while queue:
        dist, r, c = heapq.heappop(queue)
        if (r, c) not in visited:
            visited.add((r, c))
            if grid.values[r][c] in {2, 8}:
                # Backtrack to create the path
                while (r, c) != (start_r, start_c):
                    if grid.values[r][c] == 0:
                        grid.values[r][c] = 2
                    r, c = visited_from[(r, c)]
                return
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    heapq.heappush(queue, (dist + 1, nr, nc))
                    visited_from[(nr, nc)] = (r, c)

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
