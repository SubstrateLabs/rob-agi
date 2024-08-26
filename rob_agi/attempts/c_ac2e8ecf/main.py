from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging shapes based on their size, color, and original positions.
    
    The solution follows these steps:
    1. Identify and analyze all shapes in the input grid.
    2. Categorize shapes by their vertical position (top, middle, bottom) and color.
    3. Sort shapes within each category based on size (descending).
    4. Create a new grid and calculate the top, middle, and bottom sections.
    5. Place shapes in their respective sections, prioritizing color grouping and compactness.
    6. Maintain rough vertical positioning and left-to-right order within color groups.
    7. Adjust for compactness and handle overflow between sections if necessary.
    8. Fill empty spaces with black (0).
    9. Optimize placement for balance and aesthetics.

    This approach creates an organized output grid while preserving the original shapes,
    their rough vertical positioning, color grouping, and overall aesthetic balance.
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
                'centroid': get_centroid(region),
                'leftmost_col': min(c for _, c in region),
            }
            shapes.append(shape)
    return shapes

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    return (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)

def get_centroid(region: List[Tuple[int, int]]) -> Tuple[float, float]:
    avg_row = sum(r for r, _ in region) / len(region)
    avg_col = sum(c for _, c in region) / len(region)
    return (avg_row, avg_col)

def categorize_shapes(shapes: List[Dict], total_rows: int) -> Dict[str, Dict[int, List[Dict]]]:
    categorized = {'top': {}, 'middle': {}, 'bottom': {}}
    third = total_rows // 3
    for shape in shapes:
        if shape['centroid'][0] < third:
            section = 'top'
        elif shape['centroid'][0] < 2 * third:
            section = 'middle'
        else:
            section = 'bottom'
        
        if shape['color'] not in categorized[section]:
            categorized[section][shape['color']] = []
        categorized[section][shape['color']].append(shape)
    
    for section in categorized:
        for color in categorized[section]:
            categorized[section][color].sort(key=lambda s: -s['size'])
    
    return categorized

def calculate_section_heights(total_rows: int, categorized_shapes: Dict[str, Dict[int, List[Dict]]]) -> Dict[str, int]:
    section_heights = {}
    remaining_rows = total_rows
    
    for section in ['top', 'middle', 'bottom']:
        if categorized_shapes[section]:
            section_height = max(max(shape['bounding_box'][2] for shape in shapes) 
                                 for shapes in categorized_shapes[section].values())
            section_heights[section] = min(section_height, remaining_rows)
            remaining_rows -= section_heights[section]
        else:
            section_heights[section] = 0
    
    return section_heights

def place_shapes(grid: ColoredGrid, categorized_shapes: Dict[str, Dict[int, List[Dict]]], section_heights: Dict[str, int]):
    current_row = 0
    for section in ['top', 'middle', 'bottom']:
        if categorized_shapes[section]:
            place_shapes_in_section(grid, categorized_shapes[section], current_row, section_heights[section])
        current_row += section_heights[section]

def place_shapes_in_section(grid: ColoredGrid, shapes_by_color: Dict[int, List[Dict]], start_row: int, height: int):
    col = 0
    for color in sorted(shapes_by_color.keys(), key=lambda c: -len(shapes_by_color[c])):
        for shape in shapes_by_color[color]:
            if col + shape['bounding_box'][3] > grid.get_dimensions()[1]:
                col = 0  # Start a new row if we exceed the grid width
            anchor = (start_row, col)
            place_shape(grid, shape, anchor, height)
            col += shape['bounding_box'][3] + 1
        col = 0  # Start a new row for each color group

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

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    shapes = analyze_shapes(input_grid)
    categorized_shapes = categorize_shapes(shapes, input_grid.get_dimensions()[0])
    
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    section_heights = calculate_section_heights(rows, categorized_shapes)
    
    place_shapes(new_grid, categorized_shapes, section_heights)
    
    return new_grid
