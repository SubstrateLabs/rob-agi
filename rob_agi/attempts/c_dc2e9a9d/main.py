from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_dc2e9a9d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by mirroring green shapes in blue, adding sky blue elements,
    and enhancing overall balance and symmetry.

    1. Mirror large green shapes in blue
    2. Add sky blue shapes inspired by existing shapes
    3. Process medium and small shapes
    4. Fill the center if empty
    5. Balance the composition and ensure symmetry
    6. Clean up isolated cells
    """
    output_grid = input_grid.deep_copy()
    shapes = categorize_shapes(input_grid)
    occupied_cells = set((r, c) for r, row in enumerate(input_grid.values) for c, val in enumerate(row) if val != 0)
    
    center_r, center_c = output_grid.num_rows // 2, output_grid.num_cols // 2
    
    process_large_shapes(output_grid, shapes['large'], center_r, center_c, occupied_cells)
    add_sky_blue_shapes(output_grid, shapes['large'] + shapes['medium'], center_r, center_c, occupied_cells)
    process_medium_and_small_shapes(output_grid, shapes['medium'], shapes['small'], center_r, center_c, occupied_cells)
    
    fill_empty_space(output_grid, occupied_cells)
    balance_and_ensure_symmetry(output_grid, occupied_cells)
    cleanup(output_grid, occupied_cells)
    
    return output_grid

def fill_empty_space(grid: ColoredGrid, occupied_cells: Set[Tuple[int, int]]):
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    if all((r, c) not in occupied_cells for r in range(center_r - 1, center_r + 2) for c in range(center_c - 1, center_c + 2)):
        for r in range(center_r - 1, center_r + 2):
            for c in range(center_c - 1, center_c + 2):
                if (r, c) == (center_r, center_c) or (r + c) % 2 == 0:
                    grid.set_cell(r, c, 8)
                    occupied_cells.add((r, c))

def categorize_shapes(grid: ColoredGrid) -> Dict[str, List[List[Tuple[int, int]]]]:
    all_shapes = grid.find_connected_regions(3)
    categorized = {'large': [], 'medium': [], 'small': []}
    for shape in all_shapes:
        if len(shape) > 9:
            categorized['large'].append(shape)
        elif len(shape) > 4:
            categorized['medium'].append(shape)
        else:
            categorized['small'].append(shape)
    return categorized

def process_large_shapes(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    for shape in shapes:
        if not mirror_main_shape(grid, shape, center_r, center_c, occupied_cells):
            create_inspired_blue_shape(grid, shape, center_r, center_c, occupied_cells)

def create_inspired_blue_shape(grid: ColoredGrid, original_shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    new_shape = create_inspired_shape(original_shape)
    place_shape_in_empty_area(grid, new_shape, center_r, center_c, occupied_cells, color=1)

def process_medium_and_small_shapes(grid: ColoredGrid, medium_shapes: List[List[Tuple[int, int]]], small_shapes: List[List[Tuple[int, int]]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    for shape in medium_shapes:
        if should_mirror_shape(grid, shape, center_r, center_c):
            mirror_shape(grid, shape, center_r, center_c, occupied_cells, color=1)
        else:
            incorporate_into_pattern(grid, shape, occupied_cells)
    
    for shape in small_shapes:
        if len(shape) <= 2 or is_isolated(grid, shape[0][0], shape[0][1]):
            continue  # Leave very small or isolated shapes unchanged
        else:
            incorporate_into_pattern(grid, shape, occupied_cells)

def should_mirror_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int) -> bool:
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    return abs(shape_center_r - center_r) + abs(shape_center_c - center_c) > grid.num_rows // 4

def mirror_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]], color: int):
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    mirror_r = center_r + (center_r - shape_center_r)
    mirror_c = center_c + (center_c - shape_center_c)
    
    if can_place_shape(grid, shape, mirror_r, mirror_c, occupied_cells):
        place_shape(grid, shape, mirror_r, mirror_c, color, occupied_cells)

def incorporate_into_pattern(grid: ColoredGrid, shape: List[Tuple[int, int]], occupied_cells: Set[Tuple[int, int]]):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        new_shape = [(r + dr, c + dc) for r, c in shape]
        if can_place_shape(grid, new_shape, 0, 0, occupied_cells):
            place_shape(grid, new_shape, 0, 0, 1, occupied_cells)
            return

def final_check(output_grid: ColoredGrid, input_grid: ColoredGrid):
    # Implement a final check to ensure the transformation has enhanced the original design
    # while respecting its core patterns
    pass

def mirror_main_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]) -> bool:
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
    for dr, dc in directions:
        mirror_offset_r = (max(r for r, _ in shape) - min(r for r, _ in shape) + 2) * dr
        mirror_offset_c = (max(c for _, c in shape) - min(c for _, c in shape) + 2) * dc
        if can_place_shape(grid, shape, shape_center_r + mirror_offset_r, shape_center_c + mirror_offset_c, occupied_cells):
            place_shape(grid, shape, shape_center_r + mirror_offset_r, shape_center_c + mirror_offset_c, 1, occupied_cells)
            return True
    
    # If direct mirroring is not possible, try with slight modifications
    modified_shape = create_inspired_shape(shape)
    for dr, dc in directions:
        mirror_offset_r = (max(r for r, _ in modified_shape) - min(r for r, _ in modified_shape) + 2) * dr
        mirror_offset_c = (max(c for _, c in modified_shape) - min(c for _, c in modified_shape) + 2) * dc
        if can_place_shape(grid, modified_shape, shape_center_r + mirror_offset_r, shape_center_c + mirror_offset_c, occupied_cells):
            place_shape(grid, modified_shape, shape_center_r + mirror_offset_r, shape_center_c + mirror_offset_c, 1, occupied_cells)
            return True
    
    return False

def add_sky_blue_shapes(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    empty_quadrants = find_empty_quadrants(grid, center_r, center_c)
    for quadrant in empty_quadrants:
        shape_to_place = max(shapes, key=len)
        if len(shape_to_place) > 9:  # Only use larger shapes as inspiration
            new_shape = create_inspired_shape(shape_to_place)
            place_shape_in_quadrant(grid, new_shape, quadrant, center_r, center_c, occupied_cells, color=8)

def create_inspired_shape(original_shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Create a new shape inspired by the original, but not exactly the same
    min_r = min(r for r, _ in original_shape)
    min_c = min(c for _, c in original_shape)
    normalized_shape = [(r - min_r, c - min_c) for r, c in original_shape]
    
    # Simplify the shape while maintaining its general structure
    new_shape = []
    for r, c in normalized_shape:
        if (r % 2 == 0 and c % 2 == 0) or (r, c) in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            new_shape.append((r, c))
    
    return new_shape

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
def fill_empty_space(grid: ColoredGrid, occupied_cells: Set[Tuple[int, int]]):
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    if all((r, c) not in occupied_cells for r in range(center_r - 1, center_r + 2) for c in range(center_c - 1, center_c + 2)):
        for r in range(center_r - 1, center_r + 2):
            for c in range(center_c - 1, center_c + 2):
                if (r, c) == (center_r, center_c) or (r + c) % 2 == 0:
                    grid.set_cell(r, c, 8)
                    occupied_cells.add((r, c))
def process_small_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], center_r: int, center_c: int, occupied_cells: Set[Tuple[int, int]]):
    shape_center_r = sum(r for r, _ in shape) // len(shape)
    shape_center_c = sum(c for _, c in shape) // len(shape)
    
    if len(shape) <= 4 or abs(shape_center_r - center_r) + abs(shape_center_c - center_c) > grid.num_rows // 4:
        # Leave very small shapes or shapes far from center unchanged
        return
    
    # Try to incorporate into larger pattern
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dr, dc in directions:
        new_shape = [(r + dr, c + dc) for r, c in shape]
        if can_place_shape(grid, new_shape, shape_center_r, shape_center_c, occupied_cells):
            place_shape(grid, new_shape, shape_center_r, shape_center_c, 1, occupied_cells)
            return

def balance_and_ensure_symmetry(grid: ColoredGrid, occupied_cells: Set[Tuple[int, int]]):
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    color_count = {1: 0, 3: 0, 8: 0}
    left_weight, right_weight = 0, 0
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] in color_count:
                color_count[grid.values[r][c]] += 1
            if c < center_c:
                left_weight += grid.values[r][c] != 0
            else:
                right_weight += grid.values[r][c] != 0
    
    # Balance colors
    min_color = min(color_count, key=color_count.get)
    if color_count[min_color] * 2 < max(color_count.values()):
        add_balancing_shapes(grid, min_color, occupied_cells)
    
    # Ensure symmetry
    if abs(left_weight - right_weight) > 3:
        lighter_side = 'left' if left_weight < right_weight else 'right'
        add_symmetry_shapes(grid, lighter_side, occupied_cells)

def add_balancing_shapes(grid: ColoredGrid, color: int, occupied_cells: Set[Tuple[int, int]]):
    shapes = [[(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2 square
              [(0, 0), (0, 1), (0, 2), (1, 1)]]  # T shape
    
    for shape in shapes:
        for r in range(0, grid.num_rows - len(shape)):
            for c in range(0, grid.num_cols - len(shape[0])):
                if can_place_shape(grid, shape, r, c, occupied_cells):
                    place_shape(grid, shape, r, c, color, occupied_cells)
                    return

def add_symmetry_shapes(grid: ColoredGrid, side: str, occupied_cells: Set[Tuple[int, int]]):
    center_c = grid.num_cols // 2
    start_c, end_c = (0, center_c) if side == 'left' else (center_c, grid.num_cols)
    
    shapes = [[(0, 0), (0, 1)],  # 1x2 rectangle
              [(0, 0), (1, 0)]]  # 2x1 rectangle
    
    for shape in shapes:
        for r in range(0, grid.num_rows - len(shape)):
            for c in range(start_c, end_c - len(shape[0])):
                if can_place_shape(grid, shape, r, c, occupied_cells):
                    place_shape(grid, shape, r, c, 1 if side == 'left' else 8, occupied_cells)
                    return

def cleanup(grid: ColoredGrid, occupied_cells: Set[Tuple[int, int]]):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] in (1, 8) and is_isolated(grid, r, c):
                grid.values[r][c] = 0
                occupied_cells.remove((r, c))

def is_isolated(grid: ColoredGrid, r: int, c: int) -> bool:
    return all(
        grid.values[r+dr][c+dc] == 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= r+dr < grid.num_rows and 0 <= c+dc < grid.num_cols
    )
