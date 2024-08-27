from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_a934301b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a934301b challenge by identifying and preserving dominant shapes.
    
    The solution follows these steps:
    1. Identify all distinct shapes in the input grid.
    2. Divide the grid into sections (quadrants).
    3. Calculate complexity scores for each shape.
    4. Determine dominant shapes in each section and globally.
    5. Create an output grid containing only dominant shapes.
    
    A shape is considered dominant based on its complexity score,
    which takes into account size, presence of special cells (8),
    and shape irregularity.
    """
    shapes = find_shapes(input_grid)
    sections = divide_grid_into_sections(input_grid)
    scored_shapes = calculate_shape_complexity(shapes)
    dominant_shapes = find_dominant_shapes(scored_shapes, sections)
    return create_output_grid(input_grid, dominant_shapes)

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

def divide_grid_into_sections(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    return [
        (0, 0, mid_row, mid_col),
        (0, mid_col, mid_row, cols),
        (mid_row, 0, rows, mid_col),
        (mid_row, mid_col, rows, cols)
    ]

def calculate_shape_complexity(shapes: List[Set[Tuple[int, int]]]) -> List[Tuple[Set[Tuple[int, int]], int]]:
    scored_shapes = []
    for shape in shapes:
        size = len(shape)
        special_cells = sum(1 for r, c in shape if grid.get_cell(r, c) == 8)
        irregularity = calculate_irregularity(shape)
        score = size + special_cells * 2 + irregularity
        scored_shapes.append((shape, score))
    return scored_shapes

def calculate_irregularity(shape: Set[Tuple[int, int]]) -> int:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    bounding_box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
    return bounding_box_area - len(shape)

def find_dominant_shapes(scored_shapes: List[Tuple[Set[Tuple[int, int]], int]], sections: List[Tuple[int, int, int, int]]) -> List[Set[Tuple[int, int]]]:
    dominant_shapes = []
    for section in sections:
        section_shapes = [shape for shape, score in scored_shapes if shape_in_section(shape, section)]
        if section_shapes:
            max_score = max(score for shape, score in scored_shapes if shape in section_shapes)
            dominant_shapes.extend(shape for shape in section_shapes if any(score == max_score for s, score in scored_shapes if s == shape))
    return dominant_shapes

def shape_in_section(shape: Set[Tuple[int, int]], section: Tuple[int, int, int, int]) -> bool:
    min_r, min_c, max_r, max_c = section
    return any(min_r <= r < max_r and min_c <= c < max_c for r, c in shape)

def create_output_grid(input_grid: ColoredGrid, dominant_shapes: List[Set[Tuple[int, int]]]) -> ColoredGrid:
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for shape in dominant_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, input_grid.get_cell(r, c))
    return output_grid
