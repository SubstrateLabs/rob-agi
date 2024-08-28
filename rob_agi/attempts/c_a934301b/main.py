from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
import math

def solve_a934301b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a934301b challenge by identifying and preserving dominant shapes.
    
    The solution follows these steps:
    1. Create a deep copy of the input grid.
    2. Identify all distinct shapes in the grid.
    3. Divide the grid into quadrants.
    4. Calculate complexity scores for each shape based on size, special cells, irregularity, and uniqueness.
    5. Determine dominant shapes in each quadrant and globally.
    6. Create an output grid containing only dominant shapes.
    
    A shape is considered dominant based on its complexity score and uniqueness.
    The algorithm preserves shapes that are either significantly more complex than others in their quadrant,
    or unique across the entire grid.
    """
    grid_copy = input_grid.deep_copy()
    shapes = find_shapes(grid_copy)
    quadrants = divide_grid_into_quadrants(grid_copy)
    scored_shapes = calculate_shape_complexity(grid_copy, shapes)
    dominant_shapes = find_dominant_shapes(scored_shapes, quadrants)
    return create_output_grid(grid_copy, dominant_shapes)

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

def divide_grid_into_quadrants(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    return [
        (0, 0, mid_row, mid_col),
        (0, mid_col, mid_row, cols),
        (mid_row, 0, rows, mid_col),
        (mid_row, mid_col, rows, cols)
    ]

def calculate_shape_complexity(grid: ColoredGrid, shapes: List[Set[Tuple[int, int]]]) -> List[Tuple[Set[Tuple[int, int]], float]]:
    scored_shapes = []
    shape_colors = {frozenset(shape): grid.get_cell(next(iter(shape))[0], next(iter(shape))[1]) for shape in shapes}
    
    for shape in shapes:
        size = len(shape)
        special_cells = sum(1 for r, c in shape if grid.get_cell(r, c) == 8)
        irregularity = calculate_irregularity(shape)
        uniqueness = calculate_uniqueness(shape, shapes)
        
        score = size + special_cells * 2 + irregularity * 3 + uniqueness * 5
        scored_shapes.append((shape, score))
    
    return scored_shapes

def calculate_irregularity(shape: Set[Tuple[int, int]]) -> float:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    bounding_box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
    return 1 - (len(shape) / bounding_box_area)

def calculate_uniqueness(shape: Set[Tuple[int, int]], all_shapes: List[Set[Tuple[int, int]]]) -> float:
    shape_signature = frozenset((r - min(r for r, _ in shape), c - min(c for _, c in shape)) for r, c in shape)
    similar_shapes = sum(1 for other_shape in all_shapes if len(other_shape) == len(shape) and 
                         frozenset((r - min(r for r, _ in other_shape), c - min(c for _, c in other_shape)) for r, c in other_shape) == shape_signature)
    return 1 / similar_shapes

def find_dominant_shapes(scored_shapes: List[Tuple[Set[Tuple[int, int]], float]], quadrants: List[Tuple[int, int, int, int]]) -> List[Set[Tuple[int, int]]]:
    dominant_shapes = []
    global_threshold = sum(score for _, score in scored_shapes) / len(scored_shapes) * 1.5
    
    for quadrant in quadrants:
        quadrant_shapes = [shape for shape, _ in scored_shapes if shape_in_quadrant(shape, quadrant)]
        if quadrant_shapes:
            quadrant_scores = [score for shape, score in scored_shapes if shape in quadrant_shapes]
            quadrant_threshold = max(global_threshold, sum(quadrant_scores) / len(quadrant_scores) * 1.5)
            dominant_shapes.extend(shape for shape, score in scored_shapes if shape in quadrant_shapes and score >= quadrant_threshold)
    
    # Add unique shapes regardless of their quadrant
    unique_shapes = [shape for shape, score in scored_shapes if calculate_uniqueness(shape, [s for s, _ in scored_shapes]) == 1]
    dominant_shapes.extend(unique_shapes)
    
    return list(set(dominant_shapes))  # Remove duplicates

def shape_in_quadrant(shape: Set[Tuple[int, int]], quadrant: Tuple[int, int, int, int]) -> bool:
    min_r, min_c, max_r, max_c = quadrant
    return any(min_r <= r < max_r and min_c <= c < max_c for r, c in shape)

def create_output_grid(input_grid: ColoredGrid, dominant_shapes: List[Set[Tuple[int, int]]]) -> ColoredGrid:
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for shape in dominant_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, input_grid.get_cell(r, c))
    return output_grid
