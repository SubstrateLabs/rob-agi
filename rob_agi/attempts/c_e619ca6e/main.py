from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e619ca6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid expansion challenge by identifying base shapes in the input grid
    and applying predefined expansion patterns to create the output grid.
    
    The solution follows these steps:
    1. Detect all green (value 3) shapes in the input grid.
    2. For each detected shape, determine its base shape type (3x3 square, 4x2 rectangle, etc.).
    3. Apply a predefined expansion pattern for each base shape.
    4. Combine all expanded patterns into the output grid, maintaining overlaps.
    
    This approach ensures consistent expansion for identical input shapes across all test cases.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    shapes = detect_shapes(input_grid)
    for shape in shapes:
        expanded_shape = expand_shape(shape, rows, cols)
        apply_shape(output_grid, expanded_shape)
    
    return output_grid

def detect_shapes(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """Detects all green shapes in the grid and returns their bounding boxes."""
    shapes = []
    rows, cols = grid.get_dimensions()
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                top, left, bottom, right = r, c, r, c
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) not in visited and grid.get_cell(cr, cc) == 3:
                        visited.add((cr, cc))
                        bottom = max(bottom, cr)
                        right = max(right, cc)
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                shapes.append((top, left, bottom, right))
    return shapes

def expand_shape(shape: Tuple[int, int, int, int], max_rows: int, max_cols: int) -> List[Tuple[int, int]]:
    """Expands a given shape based on predefined patterns."""
    top, left, bottom, right = shape
    height, width = bottom - top + 1, right - left + 1
    
    if height == width == 3:  # 3x3 square
        pattern = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
            (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
            (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
        ]
    elif height == 2 and width == 4:  # 4x2 rectangle
        pattern = [
            (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
            (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)
        ]
    elif height == 2 and width == 5:  # 5x2 rectangle
        pattern = [
            (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5),
            (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)
        ]
    else:
        return [(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)]

    expanded = []
    center_r, center_c = (top + bottom) // 2, (left + right) // 2
    for dr, dc in pattern:
        new_r, new_c = center_r + dr, center_c + dc
        if 0 <= new_r < max_rows and 0 <= new_c < max_cols:
            expanded.append((new_r, new_c))
    return expanded

def apply_shape(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    """Applies the expanded shape to the output grid."""
    for r, c in shape:
        grid.set_cell(r, c, 3)
