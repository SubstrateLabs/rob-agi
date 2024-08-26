from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
import math
import random
import heapq

def solve_b9630600(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the b9630600 challenge by creating a cohesive and aesthetically pleasing structure
    from the original green shapes.

    The solution follows these steps:
    1. Analyze input to identify shapes and determine overall symmetry
    2. Preserve original structure by marking fixed cells
    3. Fill small hollow areas within shapes
    4. Expand shapes outwards while maintaining their form
    5. Identify connection points for each shape
    6. Create primary connections using a minimal spanning tree approach
    7. Enhance structural integrity by thickening connections and adding support
    8. Create secondary connections to improve overall structure
    9. Adjust for symmetry if present in the input
    10. Fill isolated cells and expand sparse areas
    11. Enhance aesthetics with minor details
    12. Verify connectivity of the entire structure
    13. Clean up and optimize the final structure
    14. Perform a final symmetry check and adjustment

    This approach creates a connected green structure that preserves the original shapes
    while enhancing connectivity and aesthetic appeal.
    """
    output_grid = input_grid.deep_copy()
    fixed_cells = mark_fixed_cells(output_grid)
    shapes = identify_shapes(output_grid)
    symmetry = detect_symmetry(output_grid)
    
    fill_hollow_areas(output_grid, shapes)
    expand_shapes(output_grid, shapes, fixed_cells)
    connection_points = identify_connection_points(shapes)
    create_primary_connections(output_grid, connection_points)
    enhance_structural_integrity(output_grid, shapes)
    create_secondary_connections(output_grid, shapes)
    adjust_symmetry(output_grid, symmetry)
    fill_isolated_cells_and_expand(output_grid)
    enhance_aesthetics(output_grid)
    verify_connectivity(output_grid)
    clean_up_and_optimize(output_grid, fixed_cells)
    final_symmetry_check(output_grid, symmetry)
    
    return output_grid

def detect_symmetry(grid: ColoredGrid) -> Tuple[bool, bool]:
    """Detect horizontal and vertical symmetry in the grid."""
    h_sym = all(grid.values[r] == grid.values[-r-1] for r in range(grid.num_rows // 2))
    v_sym = all(grid.values[r][c] == grid.values[r][-c-1] for r in range(grid.num_rows) for c in range(grid.num_cols // 2))
    return h_sym, v_sym

def expand_shapes(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]], fixed_cells: Set[Tuple[int, int]]):
    for shape in shapes:
        new_cells = set()
        for r, c in shape:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if (nr, nc) not in fixed_cells and grid.get_cell(nr, nc) == 0:
                        new_cells.add((nr, nc))
        for nr, nc in new_cells:
            grid.set_cell(nr, nc, 3)

def enhance_structural_integrity(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    for shape in shapes:
        if len(shape) > 10:  # Only add internal struts to larger shapes
            centroid = calculate_centroid(shape)
            corners = find_corners(shape)
            for corner in corners:
                connect_points(grid, corner, centroid)

def create_secondary_connections(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    all_points = [point for shape in shapes for point in shape]
    for i, point1 in enumerate(all_points):
        for point2 in all_points[i+1:]:
            if random.random() < 0.1:  # 10% chance to create a secondary connection
                connect_points(grid, point1, point2)

def adjust_symmetry(grid: ColoredGrid, symmetry: Tuple[bool, bool]):
    h_sym, v_sym = symmetry
    if h_sym:
        for r in range(grid.num_rows // 2):
            for c in range(grid.num_cols):
                if grid.get_cell(r, c) != grid.get_cell(grid.num_rows - 1 - r, c):
                    grid.set_cell(grid.num_rows - 1 - r, c, grid.get_cell(r, c))
    if v_sym:
        for r in range(grid.num_rows):
            for c in range(grid.num_cols // 2):
                if grid.get_cell(r, c) != grid.get_cell(r, grid.num_cols - 1 - c):
                    grid.set_cell(r, grid.num_cols - 1 - c, grid.get_cell(r, c))

def enhance_aesthetics(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                        if random.random() < 0.1:  # 10% chance to add aesthetic detail
                            grid.set_cell(nr, nc, 3)

def clean_up_and_optimize(grid: ColoredGrid, fixed_cells: Set[Tuple[int, int]]):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in fixed_cells:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)

def calculate_centroid(shape: Set[Tuple[int, int]]) -> Tuple[int, int]:
    x_sum = sum(x for x, _ in shape)
    y_sum = sum(y for _, y in shape)
    return x_sum // len(shape), y_sum // len(shape)

def find_corners(shape: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    return [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    
    fill_hollow_areas(output_grid, shapes)
    expand_shapes(output_grid, shapes, fixed_cells)
    connection_points = identify_connection_points(shapes)
    create_primary_connections(output_grid, connection_points)
    enhance_structural_integrity(output_grid, shapes)
    create_secondary_connections(output_grid, shapes)
    adjust_symmetry(output_grid, symmetry)
    fill_isolated_cells_and_expand(output_grid)
    enhance_aesthetics(output_grid)
    verify_connectivity(output_grid)
    clean_up_and_optimize(output_grid, fixed_cells)
    final_symmetry_check(output_grid, symmetry)
    
    return output_grid

def calculate_centroids(shapes: List[Set[Tuple[int, int]]]) -> List[Tuple[float, float]]:
    centroids = []
    for shape in shapes:
        if not shape:
            centroids.append((0, 0))
            continue
        x_sum = sum(x for x, _ in shape)
        y_sum = sum(y for _, y in shape)
        centroid = (x_sum / len(shape), y_sum / len(shape))
        centroids.append(centroid)
    return centroids

def fill_hollow_areas(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    for shape in shapes:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in shape and is_small_hole(grid, r, c, shape):
                    grid.set_cell(r, c, 3)

def is_small_hole(grid: ColoredGrid, r: int, c: int, shape: Set[Tuple[int, int]]) -> bool:
    if grid.get_cell(r, c) != 0:
        return False
    visited = set()
    stack = [(r, c)]
    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = curr_r + dr, curr_c + dc
            if (nr, nc) in shape:
                continue
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 0:
                stack.append((nr, nc))
    return len(visited) <= 4  # Adjust this threshold as needed

def identify_connection_points(shapes: List[Set[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    connection_points = []
    for shape in shapes:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        connection_points.extend([(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)])
    return connection_points

def create_primary_connections(grid: ColoredGrid, connection_points: List[Tuple[int, int]]):
    # Implement a minimal spanning tree algorithm here
    # For simplicity, we'll just connect all points sequentially
    for i in range(len(connection_points) - 1):
        connect_points(grid, connection_points[i], connection_points[i + 1])

def connect_points(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    r1, c1 = start
    r2, c2 = end
    while (r1, c1) != (r2, c2):
        grid.set_cell(r1, c1, 3)
        if abs(r1 - r2) > abs(c1 - c2):
            r1 += 1 if r2 > r1 else -1
        else:
            c1 += 1 if c2 > c1 else -1
    grid.set_cell(r2, c2, 3)

def fill_isolated_cells_and_expand(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 0:
                        if random.random() < 0.3:  # Adjust this probability as needed
                            grid.set_cell(nr, nc, 3)

def final_symmetry_check(grid: ColoredGrid, symmetry: Tuple[bool, bool]):
    h_sym, v_sym = symmetry
    if h_sym:
        for r in range(grid.num_rows // 2):
            for c in range(grid.num_cols):
                if grid.get_cell(r, c) != grid.get_cell(grid.num_rows - 1 - r, c):
                    grid.set_cell(grid.num_rows - 1 - r, c, grid.get_cell(r, c))
    if v_sym:
        for r in range(grid.num_rows):
            for c in range(grid.num_cols // 2):
                if grid.get_cell(r, c) != grid.get_cell(r, grid.num_cols - 1 - c):
                    grid.set_cell(r, grid.num_cols - 1 - c, grid.get_cell(r, c))

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
