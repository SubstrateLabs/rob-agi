from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_a934301b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a934301b challenge by identifying and preserving non-dominated shapes.
    
    The solution follows these steps:
    1. Identify all distinct shapes in the input grid.
    2. Analyze each shape to create a normalized representation.
    3. Compare shapes to determine dominance relationships.
    4. Create an output grid containing only non-dominated shapes.
    
    A shape is considered dominated if another shape can fully contain it
    when moved, rotated, or flipped.
    """
    shapes = find_shapes(input_grid)
    non_dominated = find_non_dominated_shapes(shapes)
    return create_output_grid(input_grid, non_dominated)

def find_shapes(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0 and (r, c) not in visited:
                shape = set()
                dfs(grid, r, c, grid.get_cell(r, c), shape, visited)
                shapes.append(shape)
    return shapes

def dfs(grid: ColoredGrid, r: int, c: int, color: int, shape: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]):
    if not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols) or grid.get_cell(r, c) != color or (r, c) in visited:
        return
    visited.add((r, c))
    shape.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, shape, visited)

def find_non_dominated_shapes(shapes: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    non_dominated = []
    for i, shape in enumerate(shapes):
        if not any(dominates(other, shape) for j, other in enumerate(shapes) if i != j):
            non_dominated.append(shape)
    return non_dominated

def dominates(shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]) -> bool:
    if len(shape1) < len(shape2):
        return False
    
    normalized1 = normalize_shape(shape1)
    normalized2 = normalize_shape(shape2)
    
    for rotation in range(4):
        rotated2 = rotate_shape(normalized2, rotation)
        if can_contain(normalized1, rotated2):
            return True
    return False

def normalize_shape(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return {(r - min_r, c - min_c) for r, c in shape}

def rotate_shape(shape: Set[Tuple[int, int]], rotation: int) -> Set[Tuple[int, int]]:
    if rotation == 0:
        return shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    if rotation == 1:
        return {(c, max_r - r) for r, c in shape}
    elif rotation == 2:
        return {(max_r - r, max_c - c) for r, c in shape}
    else:
        return {(max_c - c, r) for r, c in shape}

def can_contain(shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]) -> bool:
    max_r1 = max(r for r, _ in shape1)
    max_c1 = max(c for _, c in shape1)
    max_r2 = max(r for r, _ in shape2)
    max_c2 = max(c for _, c in shape2)
    
    for dr in range(max_r1 - max_r2 + 1):
        for dc in range(max_c1 - max_c2 + 1):
            if all((r + dr, c + dc) in shape1 for r, c in shape2):
                return True
    return False

def create_output_grid(input_grid: ColoredGrid, non_dominated: List[Set[Tuple[int, int]]]) -> ColoredGrid:
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for shape in non_dominated:
        for r, c in shape:
            output_grid.set_cell(r, c, input_grid.get_cell(r, c))
    return output_grid
