from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Set

def solve_dc2e9a9d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies all green (3) shapes.
    2. Mirrors the main green shape with blue (1) on the opposite side of the grid center.
    3. Adds sky blue (8) shapes in empty areas for balance.
    4. Processes smaller green shapes by mirroring or leaving them unchanged.
    5. Ensures no overlaps between new and existing shapes.
    """
    output_grid = input_grid.deep_copy()
    shapes = find_green_shapes(input_grid)
    occupied_cells = set((r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val != 0)
    
    main_shape = max(shapes, key=len)
    other_shapes = [shape for shape in shapes if shape != main_shape]
    
    center_r, center_c = output_grid.num_rows // 2, output_grid.num_cols // 2
    
    mirror_main_shape(output_grid, main_shape, center_r, center_c, occupied_cells)
    
    add_sky_blue_shapes(output_grid, shapes, center_r, center_c, occupied_cells)
    
    for shape in other_shapes:
        process_small_shape(output_grid, shape, center_r, center_c, occupied_cells)
    
    return output_grid

def find_green_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(3)

def mirror_main_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    
    mirror_offset_r = 2 * (center_r - shape_center_r)
    mirror_offset_c = 2 * (center_c - shape_center_c)
    
    for r, c in shape:
        new_r, new_c = r + mirror_offset_r, c + mirror_offset_c
        if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols and (new_r, new_c) not in occupied_cells:
            grid.set_cell(new_r, new_c, 1)  # Set blue mirror
            occupied_cells.add((new_r, new_c))

def add_sky_blue_shapes(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    empty_quadrants = find_empty_quadrants(grid, center_r, center_c)
    for quadrant in empty_quadrants:
        shape_to_place = max(shapes, key=len)
        place_shape_in_quadrant(grid, shape_to_place, quadrant, center_r, center_c, occupied_cells)

def process_small_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    if len(shape) <= 4:  # Leave very small shapes unchanged
        return
    
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    
    if abs(shape_center_r - center_r) + abs(shape_center_c - center_c) < grid.num_rows // 4:
        # If the shape is close to the center, mirror it
        mirror_main_shape(grid, shape, center_r, center_c, occupied_cells)
    else:
        # If the shape is far from the center, try to place a sky blue version nearby
        place_shape_nearby(grid, shape, shape_center_r, shape_center_c, occupied_cells)

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
def find_empty_quadrants(grid: ColoredGrid, center_r: int, center_c: int) -> List[Tuple[int, int, int, int]]:
    quadrants = [
        (0, 0, center_r, center_c),
        (0, center_c, center_r, grid.num_cols),
        (center_r, 0, grid.num_rows, center_c),
        (center_r, center_c, grid.num_rows, grid.num_cols)
    ]
    return [q for q in quadrants if is_quadrant_empty(grid, q)]

def is_quadrant_empty(grid: ColoredGrid, quadrant: Tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = quadrant
    return all(grid.values[r][c] == 0 for r in range(top, bottom) for c in range(left, right))

def place_shape_in_quadrant(grid: ColoredGrid, shape: List[Tuple[int, int]], quadrant: Tuple[int, int, int, int], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    top, left, bottom, right = quadrant
    shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
    
    start_r = (top + bottom - shape_height) // 2
    start_c = (left + right - shape_width) // 2
    
    if can_place_shape(grid, shape, start_r, start_c, occupied_cells):
        place_shape(grid, shape, start_r, start_c, 8, occupied_cells)

def place_shape_nearby(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
    
    for dr, dc in directions:
        start_r = center_r + dr * shape_height
        start_c = center_c + dc * shape_width
        if can_place_shape(grid, shape, start_r, start_c, occupied_cells):
            place_shape(grid, shape, start_r, start_c, 8, occupied_cells)
            return
