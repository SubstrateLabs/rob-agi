from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_55059096(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting green crosses with minimal red paths.
    
    1. Identify all green crosses in the grid.
    2. Create a graph where nodes are crosses and edges are potential connections.
    3. Find the Minimum Spanning Tree (MST) of this graph.
    4. Apply the MST to the grid by drawing red paths between connected crosses.
    
    This approach ensures a minimal continuous shape connecting the crosses.
    """
    # Step 1: Identify green crosses
    crosses = find_crosses(input_grid)
    
    # Step 2 & 3: Create graph and find MST
    mst = find_minimum_spanning_tree(crosses, input_grid)
    
    # Step 4: Apply MST to the grid
    output_grid = apply_mst_to_grid(input_grid, mst)
    
    return output_grid

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
