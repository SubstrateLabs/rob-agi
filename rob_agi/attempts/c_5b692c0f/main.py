from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5b692c0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by vertically mirroring and completing shapes.
    
    The function performs the following steps:
    1. Identifies connected regions (shapes) in the input grid.
    2. For each shape:
       a. Determines its bounding box and vertical midpoint.
       b. Creates a vertically mirrored version of the shape.
       c. Completes the shape by filling in gaps symmetrically.
    3. Places the transformed shapes onto a new grid.
    
    This results in more symmetrical and complete versions of the original shapes,
    while maintaining their relative positions and color patterns.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    for color in range(1, 10):  # Exclude black (0)
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            transformed_shape = transform_shape(input_grid, region)
            place_shape(output_grid, transformed_shape)
    
    return output_grid

def transform_shape(grid: ColoredGrid, region: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    if not region:
        return []
    
    top = min(r for r, _ in region)
    bottom = max(r for r, _ in region)
    left = min(c for _, c in region)
    right = max(c for _, c in region)
    
    midpoint = (top + bottom) // 2
    shape = []
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) in region:
                color = grid.get_cell(r, c)
                shape.append((r, c, color))
                if r != midpoint:
                    mirrored_r = 2 * midpoint - r
                    shape.append((mirrored_r, c, color))
    
    return shape

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int, int]]):
    for r, c, color in shape:
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            grid.set_cell(r, c, color)
