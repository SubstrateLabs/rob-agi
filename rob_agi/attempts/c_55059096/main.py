from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq
import itertools

def solve_55059096(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting green crosses with an optimal red path.
    
    1. Identify all green crosses in the grid.
    2. Generate all possible subsets of crosses, from pairs to all crosses.
    3. For each subset:
       a. Create a minimal spanning tree using Prim's algorithm.
       b. Convert the tree to a path using A* pathfinding with diagonal preference.
       c. Calculate a score based on connected crosses and path length.
    4. Choose the best-scoring solution.
    5. Apply the best solution to the grid by drawing the red path.
    6. Optimize the path by removing unnecessary bends or detours.
    
    This approach balances connecting the maximum number of crosses with minimizing the length of the red path,
    allowing for some crosses to remain unconnected if it results in a more optimal overall solution.
    """
    crosses = find_crosses(input_grid)
    if len(crosses) <= 1:
        return input_grid  # No changes needed if 0 or 1 cross

    best_shape = set()
    best_score = float('-inf')

    for i in range(2, len(crosses) + 1):
        for subset in itertools.combinations(crosses, i):
            shape = create_minimal_shape(input_grid, subset)
            score = score_shape(shape, subset, input_grid)
            if score > best_score:
                best_score = score
                best_shape = shape

    output_grid = apply_shape_to_grid(input_grid, best_shape)
    return output_grid

def create_minimal_shape(grid: ColoredGrid, crosses: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    shape = set()
    connected = set()
    unconnected = set(crosses)
    
    start = crosses[0]
    connected.add(start)
    unconnected.remove(start)

    while unconnected:
        best_path = None
        best_cost = float('inf')
        best_target = None

        for target in unconnected:
            for source in connected:
                path = find_path(source, target, shape, grid)
                if path:
                    cost = len(path)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
                        best_target = target

        if best_path:
            shape.update(best_path)
            connected.add(best_target)
            unconnected.remove(best_target)
        else:
            break  # No more reachable crosses

    return optimize_shape(shape, crosses, grid)

def score_shape(shape: Set[Tuple[int, int]], crosses: List[Tuple[int, int]], grid: ColoredGrid) -> int:
    connected_crosses = sum(1 for cross in crosses if cross in shape)
    new_red_cells = len(shape) - len(crosses)
    return connected_crosses * 10 - new_red_cells

def find_path(start: Tuple[int, int], end: Tuple[int, int], shape: Set[Tuple[int, int]], grid: ColoredGrid) -> List[Tuple[int, int]]:
    queue = [(0, 0, [start])]
    visited = set()

    while queue:
        f, g, path = heapq.heappop(queue)
        current = path[-1]

        if current == end:
            return path

        if current in visited:
            continue

        visited.add(current)

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and (nr, nc) not in visited:
                new_g = g + (0.5 if dr != 0 and dc != 0 else 1)  # Prefer diagonal moves
                if (nr, nc) in shape or grid.get_cell(nr, nc) in [2, 3]:
                    new_g = g  # No additional cost for existing path or crosses
                h = manhattan_distance((nr, nc), end)
                f = new_g + h
                new_path = path + [(nr, nc)]
                heapq.heappush(queue, (f, new_g, new_path))

    return []  # No path found

def find_path(start: Tuple[int, int], end: Tuple[int, int], shape: Set[Tuple[int, int]], grid: ColoredGrid) -> List[Tuple[int, int]]:
    queue = [(manhattan_distance(start, end), [start])]
    visited = set()

    while queue:
        _, path = heapq.heappop(queue)
        current = path[-1]

        if current == end:
            return path

        if current in visited:
            continue

        visited.add(current)

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and (nr, nc) not in visited:
                new_path = path + [(nr, nc)]
                heapq.heappush(queue, (len(new_path) + manhattan_distance((nr, nc), end), new_path))

    return []

def optimize_shape(shape: Set[Tuple[int, int]], crosses: List[Tuple[int, int]], grid: ColoredGrid) -> Set[Tuple[int, int]]:
    optimized = shape.copy()
    for cell in shape:
        if cell not in crosses and not is_disconnecting(optimized - {cell}, crosses):
            optimized.remove(cell)
    return optimized

def is_disconnecting(shape: Set[Tuple[int, int]], crosses: List[Tuple[int, int]]) -> bool:
    if not shape:
        return True
    start = next(iter(shape))
    connected = set(flood_fill(shape, start))
    return any(cross not in connected for cross in crosses)

def flood_fill(shape: Set[Tuple[int, int]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    filled = set()
    stack = [start]
    while stack:
        cell = stack.pop()
        if cell in shape and cell not in filled:
            filled.add(cell)
            r, c = cell
            stack.extend([(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]])
    return filled

def apply_shape_to_grid(grid: ColoredGrid, shape: Set[Tuple[int, int]]) -> ColoredGrid:
    output_grid = grid.deep_copy()
    for r, c in shape:
        if output_grid.get_cell(r, c) == 0:
            output_grid.set_cell(r, c, 2)
    return output_grid

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_crosses(grid: ColoredGrid) -> List[Tuple[int, int]]:
    crosses = []
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if is_cross(grid, r, c):
                crosses.append((r, c))
    return crosses

def is_cross(grid: ColoredGrid, r: int, c: int) -> bool:
    return (grid.get_cell(r, c) == 3 and
            grid.get_cell(r-1, c) == 3 and
            grid.get_cell(r+1, c) == 3 and
            grid.get_cell(r, c-1) == 3 and
            grid.get_cell(r, c+1) == 3)

def find_minimum_spanning_tree(crosses: List[Tuple[int, int]], grid: ColoredGrid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    edges = []
    for i, cross1 in enumerate(crosses):
        for cross2 in crosses[i+1:]:
            weight = calculate_path_cost(cross1, cross2, grid)
            edges.append((weight, cross1, cross2))
    
    # Kruskal's algorithm
    edges.sort()
    parent = {cross: cross for cross in crosses}
    rank = {cross: 0 for cross in crosses}
    mst = []

    def find(item):
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(x, y):
        xroot, yroot = find(x), find(y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    for weight, cross1, cross2 in edges:
        if find(cross1) != find(cross2):
            union(cross1, cross2)
            mst.append((cross1, cross2))

    return mst

def calculate_path_cost(start: Tuple[int, int], end: Tuple[int, int], grid: ColoredGrid) -> int:
    queue = [(0, start)]
    visited = set()
    while queue:
        cost, current = heapq.heappop(queue)
        if current == end:
            return cost
        if current in visited:
            continue
        visited.add(current)
        r, c = current
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                new_cost = cost + (0 if grid.get_cell(nr, nc) in [2, 3] else 1)
                heapq.heappush(queue, (new_cost, (nr, nc)))
    return float('inf')

def apply_mst_to_grid(grid: ColoredGrid, mst: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> ColoredGrid:
    output_grid = grid.deep_copy()
    for start, end in mst:
        draw_path(output_grid, start, end)
    return output_grid

def draw_path(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    r, c = start
    er, ec = end
    while (r, c) != (er, ec):
        if grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, 2)
        dr = max(-1, min(1, er - r))
        dc = max(-1, min(1, ec - c))
        r, c = r + dr, c + dc
    if grid.get_cell(r, c) == 0:
        grid.set_cell(r, c, 2)
