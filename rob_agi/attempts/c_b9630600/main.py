from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_b9630600(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the b9630600 challenge by creating a minimally connected structure from the original green shapes.

    The solution follows these steps:
    1. Analyze the input grid to identify green shapes and their characteristics
    2. Fill hollow shapes while preserving significant internal structures
    3. Identify connection points for each shape
    4. Create a minimal spanning structure connecting all shapes
    5. Clean up the structure by removing isolated cells and ensuring all original green cells are included
    6. Verify connectivity of the final structure
    7. Perform a final check to ensure layout and significant structures are maintained

    This approach creates a minimally connected green structure that preserves the key features
    and relative positions of the original shapes.
    """
    output_grid = input_grid.deep_copy()
    shapes = identify_shapes(output_grid)
    fill_hollow_shapes(output_grid, shapes)
    connection_points = identify_connection_points(shapes)
    create_minimal_spanning_structure(output_grid, connection_points)
    clean_up_structure(output_grid, input_grid)
    verify_connectivity(output_grid)
    
    return output_grid

def find_boundaries(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    boundaries = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 0:
                        boundaries.add((r, c))
                        break
    return boundaries

def fill_interior(grid: ColoredGrid, boundaries: Set[Tuple[int, int]]):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in boundaries and grid.get_cell(r, c) == 0:
                if is_inside_boundary(grid, r, c, boundaries):
                    grid.set_cell(r, c, 3)

def is_inside_boundary(grid: ColoredGrid, r: int, c: int, boundaries: Set[Tuple[int, int]]) -> bool:
    crossings = 0
    for i in range(c, grid.num_cols):
        if (r, i) in boundaries:
            crossings += 1
    return crossings % 2 == 1

def preserve_holes(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.get_cell(r, c) == 0 and output_grid.get_cell(r, c) == 3:
                if is_characteristic_hole(input_grid, r, c):
                    flood_fill_hole(output_grid, r, c)

def is_characteristic_hole(grid: ColoredGrid, r: int, c: int) -> bool:
    surrounding_green = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
    return surrounding_green >= 3

def flood_fill_hole(grid: ColoredGrid, r: int, c: int):
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.get_cell(r, c) == 3:
            grid.set_cell(r, c, 0)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))

def ensure_connectivity(grid: ColoredGrid):
    components = find_connected_components(grid)
    if len(components) > 1:
        main_component = max(components, key=len)
        for component in components:
            if component != main_component:
                connect_components(grid, main_component, component)

def find_connected_components(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    components = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                component = set()
                dfs(grid, r, c, component, visited)
                components.append(component)
    return components

def dfs(grid: ColoredGrid, r: int, c: int, component: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]):
    if not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols) or grid.get_cell(r, c) != 3 or (r, c) in visited:
        return
    visited.add((r, c))
    component.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, component, visited)

def connect_components(grid: ColoredGrid, comp1: Set[Tuple[int, int]], comp2: Set[Tuple[int, int]]):
    min_distance = float('inf')
    connection = None
    for r1, c1 in comp1:
        for r2, c2 in comp2:
            distance = abs(r1 - r2) + abs(c1 - c2)
            if distance < min_distance:
                min_distance = distance
                connection = ((r1, c1), (r2, c2))
    if connection:
        r1, c1 = connection[0]
        r2, c2 = connection[1]
        while (r1, c1) != (r2, c2):
            grid.set_cell(r1, c1, 3)
            if r1 < r2:
                r1 += 1
            elif r1 > r2:
                r1 -= 1
            elif c1 < c2:
                c1 += 1
            elif c1 > c2:
                c1 -= 1

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

def clean_up(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)
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
