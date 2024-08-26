from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_5b692c0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rotating, expanding, and creating symmetry in shapes.
    
    The function performs the following steps:
    1. Identifies connected regions (shapes) in the input grid.
    2. For each shape:
       a. Determines its bounding box and orientation.
       b. Rotates the shape 90 degrees clockwise.
       c. Expands the shape by filling empty cells within the bounding box.
       d. Creates symmetry by mirroring the more detailed half.
       e. Smooths edges to create more cohesive shapes.
    3. Places the transformed shapes onto a new grid.
    
    This results in rotated, expanded, and more symmetrical versions of the original shapes,
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
    
    # Determine bounding box
    top = min(r for r, _ in region)
    bottom = max(r for r, _ in region)
    left = min(c for _, c in region)
    right = max(c for _, c in region)
    
    # Rotate shape
    rotated_shape = rotate_shape(grid, region, top, left, bottom, right)
    
    # Expand shape
    expanded_shape = expand_shape(rotated_shape, top, left, bottom, right)
    
    # Create symmetry
    symmetrical_shape = create_symmetry(expanded_shape, top, left, bottom, right)
    
    # Smooth edges
    smoothed_shape = smooth_edges(symmetrical_shape, top, left, bottom, right)
    
    return smoothed_shape

def rotate_shape(grid: ColoredGrid, region: List[Tuple[int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    center_r, center_c = (top + bottom) // 2, (left + right) // 2
    rotated = []
    for r, c in region:
        new_r = center_r + (c - center_c)
        new_c = center_c - (r - center_r)
        color = grid.get_cell(r, c)
        rotated.append((new_r, new_c, color))
    return rotated

def expand_shape(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    expanded = shape.copy()
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if not any(cell[0] == r and cell[1] == c for cell in shape):
                adjacent_colors = [cell[2] for cell in shape if abs(cell[0] - r) + abs(cell[1] - c) == 1]
                if adjacent_colors:
                    most_common_color = max(set(adjacent_colors), key=adjacent_colors.count)
                    expanded.append((r, c, most_common_color))
    return expanded

def create_symmetry(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    center_c = (left + right) // 2
    left_half = [cell for cell in shape if cell[1] <= center_c]
    right_half = []
    for r, c, color in left_half:
        right_half.append((r, 2 * center_c - c, color))
    return left_half + right_half

def smooth_edges(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    smoothed = shape.copy()
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c, 0) not in shape:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc, 0) in shape]
                if len(neighbors) >= 5:
                    most_common_color = max(set(cell[2] for cell in shape if (cell[0], cell[1]) in neighbors), key=lambda x: sum(1 for cell in shape if cell[2] == x and (cell[0], cell[1]) in neighbors))
                    smoothed.append((r, c, most_common_color))
    return smoothed

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int, int]]):
    for r, c, color in shape:
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            grid.set_cell(r, c, color)
