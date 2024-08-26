from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Set

def solve_dc2e9a9d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies green (3) shapes and classifies them.
    2. For main 'P'-like shapes, creates a blue (1) mirror image on the opposite side.
    3. Creates larger sky blue (8) versions of significant shapes in available spaces.
    4. Adds complementary sky blue shapes in large empty areas, especially near the center.
    5. Processes smaller green shapes if space allows.
    6. Ensures no overlaps between new and existing shapes.
    """
    output_grid = input_grid.deep_copy()
    shapes = find_green_shapes(input_grid)
    occupied_cells = set((r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val != 0)
    
    main_shapes = [shape for shape in shapes if len(shape) >= 13]
    other_shapes = [shape for shape in shapes if len(shape) < 13]
    
    for shape in main_shapes:
        process_main_shape(output_grid, shape, occupied_cells)
    
    add_sky_blue_shapes(output_grid, main_shapes, occupied_cells)
    
    for shape in other_shapes:
        process_small_shape(output_grid, shape, occupied_cells)
    
    fill_empty_space(output_grid, occupied_cells)
    
    return output_grid

def find_green_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(3)

def process_main_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], occupied_cells: Set[Tuple[int, int]]):
    left = min(c for _, c in shape)
    right = max(c for _, c in shape)
    center_c = grid.num_cols // 2
    
    if left < center_c:
        mirror_direction = 1  # Mirror to the right
    else:
        mirror_direction = -1  # Mirror to the left
    
    for r, c in shape:
        new_c = c + mirror_direction * (abs(c - (left if mirror_direction > 0 else right)) + 1)
        if 0 <= new_c < grid.num_cols and (r, new_c) not in occupied_cells:
            grid.set_cell(r, new_c, 1)  # Set blue mirror
            occupied_cells.add((r, new_c))

def add_sky_blue_shapes(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]], occupied_cells: Set[Tuple[int, int]]):
    for shape in shapes:
        top = min(r for r, _ in shape)
        bottom = max(r for r, _ in shape)
        left = min(c for _, c in shape)
        right = max(c for _, c in shape)
        
        # Try to place sky blue shape in the center first
        center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
        if can_place_shape(grid, shape, center_r - (bottom - top) // 2, center_c - (right - left) // 2, occupied_cells):
            place_shape(grid, shape, center_r - (bottom - top) // 2, center_c - (right - left) // 2, 8, occupied_cells)
        elif bottom + (bottom - top) + 2 < grid.num_rows and can_place_shape(grid, shape, bottom + 2, left, occupied_cells):
            place_shape(grid, shape, bottom + 2, left, 8, occupied_cells)
        elif top - (bottom - top) - 2 >= 0 and can_place_shape(grid, shape, top - (bottom - top) - 2, left, occupied_cells):
            place_shape(grid, shape, top - (bottom - top) - 2, left, 8, occupied_cells)

def process_small_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], occupied_cells: Set[Tuple[int, int]]):
    if is_square(shape):
        process_square_shape(grid, shape, occupied_cells)
    else:
        # For non-square shapes, try to create a blue mirror if space allows
        process_main_shape(grid, shape, occupied_cells)

def is_square(shape: List[Tuple[int, int]]) -> bool:
    rows = set(r for r, _ in shape)
    cols = set(c for _, c in shape)
    return len(rows) == len(cols) and len(shape) == len(rows) * len(cols)

def process_square_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], occupied_cells: Set[Tuple[int, int]]):
    left = min(c for _, c in shape)
    top = min(r for r, _ in shape)
    size = int(len(shape) ** 0.5)
    
    # Try to add blue mirror to the right
    if can_place_shape(grid, shape, top, left + size + 1, occupied_cells):
        place_shape(grid, shape, top, left + size + 1, 1, occupied_cells)
    # If not possible, try to add sky blue square below
    elif can_place_shape(grid, shape, top + size + 1, left, occupied_cells):
        place_shape(grid, shape, top + size + 1, left, 8, occupied_cells)

def can_place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], start_r: int, start_c: int, occupied_cells: Set[Tuple[int, int]]) -> bool:
    return all(0 <= start_r + (r - min(r for r, _ in shape)) < grid.num_rows and
               0 <= start_c + (c - min(c for _, c in shape)) < grid.num_cols and
               (start_r + (r - min(r for r, _ in shape)), start_c + (c - min(c for _, c in shape))) not in occupied_cells
               for r, c in shape)

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], start_r: int, start_c: int, color: int, occupied_cells: Set[Tuple[int, int]]):
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    for r, c in shape:
        new_r, new_c = start_r + (r - min_r), start_c + (c - min_c)
        grid.set_cell(new_r, new_c, color)
        occupied_cells.add((new_r, new_c))

def fill_empty_space(grid: ColoredGrid, occupied_cells: Set[Tuple[int, int]]):
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    if all((r, c) not in occupied_cells for r in range(center_r - 2, center_r + 3) for c in range(center_c - 2, center_c + 3)):
        for r in range(center_r - 2, center_r + 3):
            grid.set_cell(r, center_c, 8)
            occupied_cells.add((r, center_c))
        for c in range(center_c - 2, center_c + 3):
            grid.set_cell(center_r, c, 8)
            occupied_cells.add((center_r, c))
