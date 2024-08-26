from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging shapes based on their size and original positions.
    
    The solution follows these steps:
    1. Identify and analyze all shapes in the input grid.
    2. Group shapes by their vertical position (top, middle, bottom).
    3. Sort shapes within each group based on size (descending) and leftmost column (ascending).
    4. Create a new grid and calculate the top, middle, and bottom sections.
    5. Place shapes in their respective sections, maintaining relative horizontal positioning.
    6. Handle any overflow by adjusting positions within sections.
    7. Fill empty spaces with black (0).

    This approach creates an organized output grid while preserving the original shapes,
    their relative vertical positioning, and approximate horizontal positioning.
    """
    shapes = analyze_shapes(input_grid)
    grouped_shapes = group_shapes_by_position(shapes, input_grid.get_dimensions()[0])
    
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    section_heights = calculate_section_heights(rows, grouped_shapes)
    
    place_shapes_in_sections(new_grid, grouped_shapes, section_heights)
    
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
                'leftmost_col': min(c for _, c in region),
                'top_row': min(r for r, _ in region)
            }
            shapes.append(shape)
    return shapes

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    return (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)

def group_shapes_by_position(shapes: List[Dict], total_rows: int) -> Dict[str, List[Dict]]:
    grouped_shapes = defaultdict(list)
    third = total_rows // 3
    for shape in shapes:
        if shape['top_row'] < third:
            grouped_shapes['top'].append(shape)
        elif shape['top_row'] < 2 * third:
            grouped_shapes['middle'].append(shape)
        else:
            grouped_shapes['bottom'].append(shape)
    
    for group in grouped_shapes.values():
        group.sort(key=lambda s: (-s['size'], s['leftmost_col']))
    
    return grouped_shapes

def calculate_section_heights(total_rows: int, grouped_shapes: Dict[str, List[Dict]]) -> Dict[str, int]:
    section_heights = {}
    remaining_rows = total_rows
    
    for section in ['top', 'middle', 'bottom']:
        if section in grouped_shapes:
            section_height = max(shape['bounding_box'][2] for shape in grouped_shapes[section])
            section_heights[section] = min(section_height, remaining_rows)
            remaining_rows -= section_heights[section]
        else:
            section_heights[section] = 0
    
    return section_heights

def place_shapes_in_sections(grid: ColoredGrid, grouped_shapes: Dict[str, List[Dict]], section_heights: Dict[str, int]):
    current_row = 0
    for section in ['top', 'middle', 'bottom']:
        if section in grouped_shapes:
            place_shapes_in_row(grid, grouped_shapes[section], current_row, section_heights[section])
        current_row += section_heights[section]

def place_shapes_in_row(grid: ColoredGrid, shapes: List[Dict], start_row: int, height: int):
    col = 0
    for shape in shapes:
        if col + shape['bounding_box'][3] > grid.get_dimensions()[1]:
            col = 0  # Start a new row if we exceed the grid width
        anchor = (start_row, col)
        place_shape(grid, shape, anchor, height)
        col += shape['bounding_box'][3] + 1

def place_shape(grid: ColoredGrid, shape: Dict, anchor: Tuple[int, int], max_height: int) -> bool:
    rows, cols = grid.get_dimensions()
    shape_height = min(shape['bounding_box'][2], max_height)
    
    for dr in range(shape_height):
        for dc in range(shape['bounding_box'][3]):
            r, c = anchor[0] + dr, anchor[1] + dc
            if r >= rows or c >= cols or grid.get_cell(r, c) != 0:
                return False
    
    for r, c in shape['cells']:
        dr, dc = r - shape['bounding_box'][0], c - shape['bounding_box'][1]
        if dr < shape_height:
            grid.set_cell(anchor[0] + dr, anchor[1] + dc, shape['color'])
    return True
