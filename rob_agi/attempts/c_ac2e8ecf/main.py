from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging shapes based on their size and original positions.
    
    The solution follows these steps:
    1. Identify and analyze all shapes in the input grid.
    2. Sort shapes based on size (descending) and leftmost column (ascending).
    3. Create a new grid and calculate the midpoint row.
    4. Place shapes in the top section, starting from the top-left corner.
    5. Place remaining shapes in the bottom section, starting from the bottom-left corner.
    6. Handle any overflow by placing remaining shapes in available spaces.
    7. Fill empty spaces with black (0).

    This approach creates an organized output grid while preserving the original shapes
    and their relative horizontal positioning.
    """
    shapes = analyze_shapes(input_grid)
    shapes.sort(key=lambda s: (-s['size'], s['leftmost_col']))
    
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    midpoint = rows // 2

    place_shapes_top(new_grid, shapes, midpoint)
    place_shapes_bottom(new_grid, shapes, midpoint)
    handle_overflow(new_grid, shapes)
    
    return new_grid

def analyze_shapes(grid: ColoredGrid) -> List[Dict]:
    shapes = []
    for color in range(1, 10):  # Exclude black (0)
        regions = grid.find_connected_regions(color)
        for region in regions:
            shape = {
                'color': color,
                'size': len(region),
                'bounding_box': get_bounding_box(region),
                'cells': region,
                'leftmost_col': min(c for _, c in region)
            }
            shapes.append(shape)
    return shapes

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    return (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)

def place_shapes_top(grid: ColoredGrid, shapes: List[Dict], midpoint: int):
    row, col = 0, 0
    for shape in shapes[:]:
        if row >= midpoint:
            break
        if place_shape(grid, shape, (row, col)):
            shapes.remove(shape)
            col += shape['bounding_box'][3] + 1
            if col + shape['bounding_box'][3] >= grid.get_dimensions()[1]:
                row += 1
                col = 0

def place_shapes_bottom(grid: ColoredGrid, shapes: List[Dict], midpoint: int):
    row, col = grid.get_dimensions()[0] - 1, 0
    for shape in shapes[:]:
        if row < midpoint:
            break
        if place_shape(grid, shape, (row - shape['bounding_box'][2] + 1, col)):
            shapes.remove(shape)
            col += shape['bounding_box'][3] + 1
            if col + shape['bounding_box'][3] >= grid.get_dimensions()[1]:
                row -= 1
                col = 0

def handle_overflow(grid: ColoredGrid, shapes: List[Dict]):
    for shape in shapes:
        for row in range(grid.get_dimensions()[0]):
            for col in range(grid.get_dimensions()[1]):
                if place_shape(grid, shape, (row, col)):
                    break
            if shape not in shapes:
                break

def place_shape(grid: ColoredGrid, shape: Dict, anchor: Tuple[int, int]) -> bool:
    rows, cols = grid.get_dimensions()
    for dr in range(shape['bounding_box'][2]):
        for dc in range(shape['bounding_box'][3]):
            r, c = anchor[0] + dr, anchor[1] + dc
            if r >= rows or c >= cols or grid.get_cell(r, c) != 0:
                return False
    
    for r, c in shape['cells']:
        dr, dc = r - shape['bounding_box'][0], c - shape['bounding_box'][1]
        grid.set_cell(anchor[0] + dr, anchor[1] + dc, shape['color'])
    return True
