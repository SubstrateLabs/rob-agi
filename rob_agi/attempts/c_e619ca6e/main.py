from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e619ca6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid expansion challenge by identifying green shapes in the input grid
    and applying a complex expansion pattern.
    
    The solution follows these steps:
    1. Detect all green (value 3) shapes in the input grid.
    2. Expand original shapes based on their dimensions.
    3. Add additional shapes around the expanded ones in two rounds.
    4. Merge overlapping shapes after each round of additions.
    5. Ensure the final structure stays within the original grid boundaries.
    
    This approach creates a branching structure that expands from the original shapes,
    consistently handling different initial configurations while maintaining symmetry
    and patterns observed in the example outputs.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    shapes = detect_shapes(input_grid)
    expanded_shapes = [expand_shape(shape, rows, cols) for shape in shapes]
    
    for shape in expanded_shapes:
        apply_shape(output_grid, shape)
    
    add_additional_shapes(output_grid, expanded_shapes)
    merge_shapes(output_grid)
    add_additional_shapes(output_grid, detect_shapes(output_grid))
    merge_shapes(output_grid)
    
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
    """Expands a given shape based on its dimensions."""
    top, left, bottom, right = shape
    height, width = bottom - top + 1, right - left + 1
    
    if height == width == 3:  # 3x3 square
        expansion = 1
    elif (height == 2 and width == 4) or (height == 4 and width == 2):  # 2x4 or 4x2 rectangle
        expansion = 1
    elif (height == 2 and width == 5) or (height == 5 and width == 2):  # 5x2 or 2x5 rectangle
        expansion = 1
    else:
        expansion = 0
    
    expanded = []
    for r in range(max(0, top - expansion), min(max_rows, bottom + expansion + 1)):
        for c in range(max(0, left - expansion), min(max_cols, right + expansion + 1)):
            expanded.append((r, c))
    return expanded

def apply_shape(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    """Applies the expanded shape to the output grid."""
    for r, c in shape:
        grid.set_cell(r, c, 3)

def add_additional_shapes(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]]):
    """Adds additional shapes around the existing ones."""
    rows, cols = grid.get_dimensions()
    for shape in shapes:
        top = min(r for r, _ in shape)
        bottom = max(r for r, _ in shape)
        left = min(c for _, c in shape)
        right = max(c for _, c in shape)
        
        # Add 3x3 squares at corners
        corner_squares = [
            (top - 2, left - 2), (top - 2, right), (bottom, left - 2), (bottom, right)
        ]
        for r, c in corner_squares:
            if 0 <= r < rows - 2 and 0 <= c < cols - 2:
                for dr in range(3):
                    for dc in range(3):
                        grid.set_cell(r + dr, c + dc, 3)
        
        # Add rectangles along sides
        side_rectangles = [
            (top - 1, left - 1, top + 1, left), (top - 1, right + 1, top + 1, right + 2),  # Vertical
            (top - 1, left, top, right), (bottom + 1, left, bottom + 2, right)  # Horizontal
        ]
        for t, l, b, r in side_rectangles:
            if 0 <= t < rows and 0 <= l < cols and b < rows and r < cols:
                for rr in range(t, b + 1):
                    for cc in range(l, r + 1):
                        grid.set_cell(rr, cc, 3)

def merge_shapes(grid: ColoredGrid):
    """Merges overlapping shapes in the grid."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 3:
                        grid.set_cell((r + nr) // 2, (c + nc) // 2, 3)
