from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_e7639916(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e7639916 challenge by connecting sky-colored (8) cells with blue (1) lines.
    
    The solution follows these steps:
    1. Identify all sky (8) cells in the input grid.
    2. If fewer than 2 sky cells, return the input grid unchanged.
    3. Create a minimum spanning tree connecting all sky cells.
    4. Close the shape by connecting leaf nodes.
    5. Draw blue (1) lines between connected sky cells.
    
    Returns a new ColoredGrid with the solution.
    """
    # Step 1: Identify sky cells
    sky_cells = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid[r][c] == 8]
    
    # Step 2: Check if there are enough sky cells
    if len(sky_cells) < 2:
        return input_grid
    
    # Step 3: Create a minimum spanning tree
    edges = kruskal_mst(sky_cells)
    
    # Step 4: Close the shape
    edges = close_shape(sky_cells, edges)
    
    # Step 5: Draw blue lines
    output_grid = input_grid.deep_copy()
    for start, end in edges:
        draw_line(output_grid, start, end)
    
    return output_grid

def kruskal_mst(points: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Compute the minimum spanning tree using Kruskal's algorithm."""
    edges = [(a, b, manhattan_distance(a, b)) for i, a in enumerate(points) for b in points[i+1:]]
    edges.sort(key=lambda x: x[2])
    
    parent = {p: p for p in points}
    rank = {p: 0 for p in points}
    
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
    
    mst = []
    for u, v, _ in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v))
    
    return mst

def close_shape(points: List[Tuple[int, int]], edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Close the shape by connecting leaf nodes."""
    degree = {p: 0 for p in points}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    leaves = [p for p in points if degree[p] == 1]
    while len(leaves) > 1:
        u, v = min(((u, v) for u in leaves for v in leaves if u != v), key=lambda x: manhattan_distance(x[0], x[1]))
        edges.append((u, v))
        leaves.remove(u)
        leaves.remove(v)
    
    return edges

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Draw a blue line between two points using Bresenham's algorithm."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if grid[y0][x0] != 8:  # Don't overwrite sky cells
            grid.values[y0][x0] = 1  # Set to blue
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
