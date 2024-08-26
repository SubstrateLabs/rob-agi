from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
import math
import random

def solve_b9630600(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the b9630600 challenge by creating a cohesive and aesthetically pleasing structure
    from the original green shapes.

    The solution follows these steps:
    1. Preserve original structure and mark fixed cells
    2. Analyze input to identify shapes, calculate centroids, and detect symmetry
    3. Expand shapes outwards where possible
    4. Create primary connections between shapes
    5. Fill hollow shapes and add internal structure
    6. Enhance structural integrity
    7. Create secondary connections
    8. Check and adjust for symmetry
    9. Fill isolated cells and expand sparse areas
    10. Enhance aesthetics
    11. Perform final connection check
    12. Clean up and optimize the structure

    This approach creates a connected green structure that preserves the original shapes
    while adding aesthetic elements and maintaining symmetry where possible.
    """
    output_grid = input_grid.deep_copy()
    fixed_cells = mark_fixed_cells(output_grid)
    shapes = identify_shapes(output_grid)
    centroids = calculate_centroids(shapes)
    symmetry = detect_symmetry(output_grid)
    
    expand_shapes(output_grid, shapes, fixed_cells)
    create_primary_connections(output_grid, shapes, centroids)
    fill_hollow_shapes(output_grid, shapes)
    enhance_structural_integrity(output_grid, shapes)
    create_secondary_connections(output_grid, shapes)
    adjust_symmetry(output_grid, symmetry)
    fill_and_expand(output_grid)
    enhance_aesthetics(output_grid)
    verify_connectivity(output_grid)
    clean_up_and_optimize(output_grid, fixed_cells)
    
    return output_grid

def mark_fixed_cells(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) == 3}

def identify_shapes(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    shapes = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                shape = set()
                dfs(grid, r, c, shape, visited)
                shapes.append(shape)
    return shapes

def dfs(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]):
    if not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols) or grid.get_cell(r, c) != 3 or (r, c) in visited:
        return
    visited.add((r, c))
    shape.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, shape, visited)

def create_graph(shapes: List[Set[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    graph = {}
    for i, shape1 in enumerate(shapes):
        graph[i] = []
        for j, shape2 in enumerate(shapes):
            if i != j:
                dist = min(manhattan_distance(p1, p2) for p1 in shape1 for p2 in shape2)
                graph[i].append((j, dist))
    return graph

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_minimal_spanning_tree(graph: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    mst = []
    visited = set()
    start_node = 0
    pq = [(0, start_node, start_node)]
    
    while pq and len(visited) < len(graph):
        weight, node, parent = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            if node != parent:
                mst.append((parent, node))
            for neighbor, edge_weight in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, neighbor, node))
    
    return mst

def connect_shapes(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]], mst: List[Tuple[int, int]]):
    for shape1_idx, shape2_idx in mst:
        shape1 = shapes[shape1_idx]
        shape2 = shapes[shape2_idx]
        p1, p2 = min((p1, p2) for p1 in shape1 for p2 in shape2 if manhattan_distance(p1, p2) == min(manhattan_distance(p1, p2) for p1 in shape1 for p2 in shape2))
        connect_points(grid, p1, p2)

def connect_points(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    r1, c1 = start
    r2, c2 = end
    while (r1, c1) != (r2, c2):
        grid.set_cell(r1, c1, 3)
        if abs(r1 - r2) > abs(c1 - c2):
            r1 += 1 if r2 > r1 else -1
        else:
            c1 += 1 if c2 > c1 else -1

def incorporate_isolated_cells(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    all_shape_cells = set.union(*shapes)
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in all_shape_cells:
                nearest_shape_cell = min(all_shape_cells, key=lambda p: manhattan_distance((r, c), p))
                connect_points(grid, (r, c), nearest_shape_cell)

def selective_fill_hollow_areas(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    for shape in shapes:
        fill_shape_interior(grid, shape)

def fill_shape_interior(grid: ColoredGrid, shape: Set[Tuple[int, int]]):
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if (r, c) not in shape and is_inside_shape(grid, r, c, shape):
                if not is_significant_hole(grid, r, c, shape):
                    grid.set_cell(r, c, 3)

def is_inside_shape(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]]) -> bool:
    crossings = 0
    for i in range(c, grid.num_cols):
        if (r, i) in shape:
            crossings += 1
    return crossings % 2 == 1

def is_significant_hole(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]]) -> bool:
    if grid.get_cell(r, c) != 0:
        return False
    hole = set()
    dfs_hole(grid, r, c, hole, shape)
    return len(hole) > 1

def dfs_hole(grid: ColoredGrid, r: int, c: int, hole: Set[Tuple[int, int]], shape: Set[Tuple[int, int]]):
    if (r, c) in hole or (r, c) in shape:
        return
    if grid.get_cell(r, c) != 0:
        return
    hole.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            dfs_hole(grid, nr, nc, hole, shape)

def optimize_connections(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)

def verify_connectivity(grid: ColoredGrid):
    green_cells = [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) == 3]
    if not green_cells:
        return
    connected = set()
    dfs(grid, green_cells[0][0], green_cells[0][1], connected, set())
    for r, c in green_cells:
        if (r, c) not in connected:
            grid.set_cell(r, c, 0)

def final_adjustments(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if input_grid.get_cell(r, c) == 3:
                output_grid.set_cell(r, c, 3)
    maintain_symmetry(output_grid)

def maintain_symmetry(grid: ColoredGrid):
    # Vertical symmetry
    for c in range(grid.num_cols // 2):
        for r in range(grid.num_rows):
            if grid.get_cell(r, c) != grid.get_cell(r, grid.num_cols - 1 - c):
                if grid.get_cell(r, c) == 3:
                    grid.set_cell(r, grid.num_cols - 1 - c, 3)
                else:
                    grid.set_cell(r, c, 3)

    # Horizontal symmetry
    for r in range(grid.num_rows // 2):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) != grid.get_cell(grid.num_rows - 1 - r, c):
                if grid.get_cell(r, c) == 3:
                    grid.set_cell(grid.num_rows - 1 - r, c, 3)
                else:
                    grid.set_cell(r, c, 3)
def identify_shapes(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    shapes = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                shape = set()
                dfs(grid, r, c, shape, visited)
                shapes.append(shape)
    return shapes

def fill_hollow_shapes(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    for shape in shapes:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in shape and is_inside_shape(grid, r, c, shape):
                    if not is_significant_hole(grid, r, c, shape):
                        grid.set_cell(r, c, 3)

def is_inside_shape(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]]) -> bool:
    crossings = 0
    for i in range(c, grid.num_cols):
        if (r, i) in shape:
            crossings += 1
    return crossings % 2 == 1

def is_significant_hole(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]]) -> bool:
    if grid.get_cell(r, c) != 0:
        return False
    hole = set()
    dfs_hole(grid, r, c, hole, shape)
    return len(hole) > 1

def dfs_hole(grid: ColoredGrid, r: int, c: int, hole: Set[Tuple[int, int]], shape: Set[Tuple[int, int]]):
    if (r, c) in hole or (r, c) in shape:
        return
    if grid.get_cell(r, c) != 0:
        return
    hole.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            dfs_hole(grid, nr, nc, hole, shape)

def identify_connection_points(shapes: List[Set[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    points = []
    for shape in shapes:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        points.extend([(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)])
    return points

def create_minimal_spanning_structure(grid: ColoredGrid, points: List[Tuple[int, int]]):
    if not points:
        return
    connected = {points[0]}
    unconnected = set(points[1:])
    while unconnected:
        start, end = min(((s, e) for s in connected for e in unconnected), key=lambda x: manhattan_distance(*x))
        connect_points(grid, start, end)
        connected.add(end)
        unconnected.remove(end)

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def connect_points(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    r1, c1 = start
    r2, c2 = end
    while (r1, c1) != (r2, c2):
        grid.set_cell(r1, c1, 3)
        if abs(r1 - r2) > abs(c1 - c2):
            r1 += 1 if r2 > r1 else -1
        else:
            c1 += 1 if c2 > c1 else -1

def clean_up_structure(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if input_grid.get_cell(r, c) == 3:
                output_grid.set_cell(r, c, 3)
    remove_isolated_cells(output_grid)

def remove_isolated_cells(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors == 0:
                    grid.set_cell(r, c, 0)

def verify_connectivity(grid: ColoredGrid):
    green_cells = [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) == 3]
    if not green_cells:
        return
    connected = set()
    dfs(grid, green_cells[0][0], green_cells[0][1], connected, set())
    for r, c in green_cells:
        if (r, c) not in connected:
            grid.set_cell(r, c, 0)
