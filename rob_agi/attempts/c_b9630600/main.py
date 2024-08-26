from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_b9630600(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the b9630600 challenge by expanding and connecting green shapes.
    
    The solution follows these steps:
    1. Identify distinct green shapes
    2. Expand shapes along their structural elements
    3. Connect shapes using the shortest path
    4. Fill enclosed spaces
    5. Adjust for symmetry and clean up
    
    This approach maintains the original shapes while creating a single
    connected green structure that follows the grid's inherent patterns.
    """
    output_grid = input_grid.deep_copy()
    shapes = find_shapes(output_grid)
    
    for shape in shapes:
        expand_shape(output_grid, shape)
    
    connect_shapes(output_grid, shapes)
    fill_enclosed_spaces(output_grid)
    clean_up(output_grid)
    
    return output_grid

def find_shapes(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
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

def expand_shape(grid: ColoredGrid, shape: Set[Tuple[int, int]]):
    edges = find_edges(shape)
    for r, c in edges:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 0:
                grid.set_cell(nr, nc, 3)
                shape.add((nr, nc))

def find_edges(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    edges = set()
    for r, c in shape:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if (r + dr, c + dc) not in shape:
                edges.add((r, c))
                break
    return edges

def connect_shapes(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]):
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            connect_two_shapes(grid, shapes[i], shapes[j])

def connect_two_shapes(grid: ColoredGrid, shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]):
    min_distance = float('inf')
    connection = None
    
    for r1, c1 in shape1:
        for r2, c2 in shape2:
            distance = abs(r1 - r2) + abs(c1 - c2)
            if distance < min_distance:
                min_distance = distance
                connection = ((r1, c1), (r2, c2))
    
    if connection:
        r1, c1 = connection[0]
        r2, c2 = connection[1]
        while (r1, c1) != (r2, c2):
            if r1 < r2:
                r1 += 1
            elif r1 > r2:
                r1 -= 1
            elif c1 < c2:
                c1 += 1
            elif c1 > c2:
                c1 -= 1
            grid.set_cell(r1, c1, 3)

def fill_enclosed_spaces(grid: ColoredGrid):
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                enclosed_space = set()
                if is_enclosed(grid, r, c, enclosed_space, visited):
                    for er, ec in enclosed_space:
                        grid.set_cell(er, ec, 3)

def is_enclosed(grid: ColoredGrid, r: int, c: int, space: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> bool:
    if not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols):
        return False
    if grid.get_cell(r, c) == 3:
        return True
    if (r, c) in visited:
        return True
    
    visited.add((r, c))
    space.add((r, c))
    
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        if not is_enclosed(grid, r + dr, c + dc, space, visited):
            return False
    
    return True

def clean_up(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)
