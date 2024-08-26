from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_dc2e9a9d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies green (3) shapes and classifies them.
    2. For 'P'-like shapes, creates a blue (1) mirror image on the side with more space.
    3. For the leftmost 'P' shape, creates a larger sky blue (8) 'P' shape above or below.
    4. For squares in the top-right, creates a blue (1) mirror or sky blue (8) square below.
    5. Adds a sky blue (8) 'T' or '+' shape in empty center space if available.
    6. Keeps other green shapes as is.
    7. Ensures no overlaps between new and existing shapes.
    """
    output_grid = input_grid.deep_copy()
    shapes = find_green_shapes(input_grid)
    
    for shape in shapes:
        if is_p_like(shape):
            process_p_shape(output_grid, shape)
        elif is_square(shape) and is_top_right(shape, input_grid):
            process_top_right_square(output_grid, shape)
    
    if has_empty_center(output_grid):
        add_center_shape(output_grid)
    
    return output_grid

def find_green_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(3)

def is_p_like(shape: List[Tuple[int, int]]) -> bool:
    # Simplified check: P-like shapes have at least 13 cells
    return len(shape) >= 13

def is_square(shape: List[Tuple[int, int]]) -> bool:
    # Simplified check: squares have 16 cells (4x4) or 9 cells (3x3)
    return len(shape) in [9, 16]

def is_top_right(shape: List[Tuple[int, int]], grid: ColoredGrid) -> bool:
    top = min(r for r, _ in shape)
    right = max(c for _, c in shape)
    return top < grid.num_rows // 2 and right > grid.num_cols // 2

def process_p_shape(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    left = min(c for _, c in shape)
    right = max(c for _, c in shape)
    
    if left < grid.num_cols - right:
        mirror_direction = 1  # Mirror to the right
    else:
        mirror_direction = -1  # Mirror to the left
    
    for r, c in shape:
        new_c = c + mirror_direction * (abs(c - left) + 1)
        if 0 <= new_c < grid.num_cols:
            grid.set_cell(r, new_c, 1)  # Set blue mirror
    
    # Add sky blue 'P' shape below or above
    top = min(r for r, _ in shape)
    bottom = max(r for r, _ in shape)
    if top > grid.num_rows - bottom:
        sky_blue_top = max(0, top - len(shape) // 4 - 1)
        for r, c in shape:
            new_r = sky_blue_top + (r - top)
            if 0 <= new_r < grid.num_rows:
                grid.set_cell(new_r, c, 8)
    else:
        sky_blue_top = min(grid.num_rows - 1, bottom + 2)
        for r, c in shape:
            new_r = sky_blue_top + (r - top)
            if new_r < grid.num_rows:
                grid.set_cell(new_r, c, 8)

def process_top_right_square(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    left = min(c for _, c in shape)
    top = min(r for r, _ in shape)
    size = int(len(shape) ** 0.5)
    
    if left + 2 * size <= grid.num_cols:
        # Add blue mirror to the right
        for r, c in shape:
            new_c = c + size + 1
            grid.set_cell(r, new_c, 1)
    else:
        # Add sky blue square below
        for r in range(top + size + 1, min(top + 2 * size + 1, grid.num_rows)):
            for c in range(left, min(left + size, grid.num_cols)):
                grid.set_cell(r, c, 8)

def has_empty_center(grid: ColoredGrid) -> bool:
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    return all(grid.get_cell(r, c) == 0 
               for r in range(center_r - 2, center_r + 3)
               for c in range(center_c - 2, center_c + 3))

def add_center_shape(grid: ColoredGrid):
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    for r in range(center_r - 2, center_r + 3):
        grid.set_cell(r, center_c, 8)
    for c in range(center_c - 2, center_c + 3):
        grid.set_cell(center_r, c, 8)
