from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_3490cc26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting sky blue (8) squares with orange (7) paths.
    
    The solution follows these steps:
    1. Identify all 2x2 sky blue squares in the input grid.
    2. Determine the bounding box of all sky blue squares.
    3. Create a graph representation within the bounding box.
    4. Find the minimum spanning tree connecting all sky blue squares.
    5. Convert the tree to an orange path.
    6. Optimize the path by removing unnecessary branches and straightening connections.
    7. Ensure adjacent sky blue squares are directly connected.
    8. Verify connectivity of all sky blue squares.
    9. Clean up any stray orange cells and preserve original colors.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with orange paths connecting sky blue squares.
    """
    output_grid = input_grid.deep_copy()
    sky_blue_squares = find_sky_blue_squares(output_grid)
    
    if not sky_blue_squares:
        return output_grid  # No sky blue squares to connect
    
    bounding_box = get_bounding_box(sky_blue_squares)
    graph = create_graph(output_grid, bounding_box)
    mst = find_minimum_spanning_tree(graph, sky_blue_squares)
    apply_mst_to_grid(output_grid, mst)
    optimize_path(output_grid, sky_blue_squares)
    connect_adjacent_squares(output_grid, sky_blue_squares)
    verify_connectivity(output_grid, sky_blue_squares)
    cleanup_and_preserve(output_grid, input_grid, bounding_box)
    
    return output_grid

def find_sky_blue_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find all 2x2 sky blue squares in the grid."""
    squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2)):
                squares.append((r, c))
    return squares

def get_bounding_box(squares: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Determine the bounding box of all sky blue squares."""
    if not squares:
        return (0, 0, 0, 0)
    min_r = min(sq[0] for sq in squares)
    max_r = max(sq[0] for sq in squares)
    min_c = min(sq[1] for sq in squares)
    max_c = max(sq[1] for sq in squares)
    return (min_r, min_c, max_r + 1, max_c + 1)  # +1 to include the full 2x2 square

def create_graph(grid: ColoredGrid, bbox: Tuple[int, int, int, int]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Create a graph representation within the bounding box."""
    min_r, min_c, max_r, max_c = bbox
    graph = {}
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid.get_cell(r, c) in [0, 7, 8]:
                graph[(r, c)] = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if min_r <= nr <= max_r and min_c <= nc <= max_c and grid.get_cell(nr, nc) in [0, 7, 8]:
                        graph[(r, c)].append((nr, nc))
    return graph

def find_minimum_spanning_tree(graph: Dict[Tuple[int, int], List[Tuple[int, int]]], squares: List[Tuple[int, int]]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find the minimum spanning tree connecting all sky blue squares using Kruskal's algorithm."""
    edges = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edges.add(tuple(sorted([node, neighbor])))
    
    parent = {node: node for node in graph}
    rank = {node: 0 for node in graph}
    
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    
    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        if root1 != root2:
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1
    
    mst = set()
    for edge in sorted(edges, key=lambda e: (e[0][0] - e[1][0])**2 + (e[0][1] - e[1][1])**2):
        if find(edge[0]) != find(edge[1]):
            union(edge[0], edge[1])
            mst.add(edge)
    
    return mst

def apply_mst_to_grid(grid: ColoredGrid, mst: Set[Tuple[Tuple[int, int], Tuple[int, int]]]):
    """Apply the minimum spanning tree to the grid by setting cells to orange."""
    for edge in mst:
        for node in edge:
            if grid.get_cell(node[0], node[1]) == 0:
                grid.set_cell(node[0], node[1], 7)

def optimize_path(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Optimize the orange path by removing unnecessary branches and straightening connections."""
    rows, cols = grid.get_dimensions()
    
    def dfs(r, c, parent):
        if grid.get_cell(r, c) != 7:
            return False
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != parent:
                if grid.get_cell(nr, nc) == 8 or dfs(nr, nc, (r, c)):
                    return True
        if (r, c) not in squares:
            grid.set_cell(r, c, 0)
        return False
    
    for r, c in squares:
        dfs(r, c, None)

def connect_adjacent_squares(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Ensure adjacent sky blue squares are directly connected."""
    for i, (r1, c1) in enumerate(squares):
        for r2, c2 in squares[i+1:]:
            if abs(r1 - r2) <= 2 and abs(c1 - c2) <= 2:
                for r in range(min(r1, r2), max(r1, r2) + 2):
                    for c in range(min(c1, c2), max(c1, c2) + 2):
                        if grid.get_cell(r, c) == 0:
                            grid.set_cell(r, c, 7)

def verify_connectivity(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Verify that all sky blue squares are connected through the orange path."""
    if not squares:
        return
    
    visited = set()
    stack = [squares[0]]
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited:
            visited.add((r, c))
            for dr in range(2):
                for dc in range(2):
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr + direction[0], c + dc + direction[1]
                        if grid.get_cell(nr, nc) == 7:
                            stack.append((nr, nc))
    
    assert len(visited) == len(squares), "Not all sky blue squares are connected"

def cleanup_and_preserve(output_grid: ColoredGrid, input_grid: ColoredGrid, bbox: Tuple[int, int, int, int]):
    """Clean up stray orange cells and preserve original non-black colors."""
    min_r, min_c, max_r, max_c = bbox
    rows, cols = output_grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 7 and not (min_r <= r <= max_r and min_c <= c <= max_c):
                output_grid.set_cell(r, c, 0)
            elif input_grid.get_cell(r, c) != 0 and output_grid.get_cell(r, c) != 7:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))
