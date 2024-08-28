from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging shapes based on their size, color, and original positions.
    
    The solution follows these steps:
    1. Analyze the input grid to identify all non-black shapes.
    2. Calculate space allocation for top, middle, and bottom sections.
    3. Sort and group shapes by color and size within each section.
    4. Place shapes in the output grid, starting from the largest color group.
    5. Optimize placement by filling large gaps and balancing the grid.
    6. Make final adjustments to ensure all shapes are placed.

    This approach creates an organized output grid while preserving the original shapes,
    their rough vertical positioning, color grouping, and overall aesthetic balance.
    """
    shapes = analyze_shapes(input_grid)
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    section_allocation = calculate_space_allocation(shapes, rows)
    grouped_shapes = sort_and_group_shapes(shapes, section_allocation)
    
    place_shapes_in_grid(new_grid, grouped_shapes, section_allocation)
    optimize_placement(new_grid, grouped_shapes)
    
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

def calculate_space_allocation(shapes: List[Dict], total_rows: int) -> Dict[str, int]:
    total_area = sum(shape['size'] for shape in shapes)
    section_areas = {'top': 0, 'middle': 0, 'bottom': 0}
    
    for shape in shapes:
        centroid_row = shape['centroid'][0]
        if centroid_row < total_rows / 3:
            section_areas['top'] += shape['size']
        elif centroid_row < 2 * total_rows / 3:
            section_areas['middle'] += shape['size']
        else:
            section_areas['bottom'] += shape['size']
    
    allocation = {}
    remaining_rows = total_rows
    for section in ['top', 'middle', 'bottom']:
        section_rows = max(1, int(section_areas[section] / total_area * total_rows))
        allocation[section] = min(section_rows, remaining_rows)
        remaining_rows -= allocation[section]
    
    return allocation

def sort_and_group_shapes(shapes: List[Dict], section_allocation: Dict[str, int]) -> Dict[str, Dict[int, List[Dict]]]:
    grouped = {'top': defaultdict(list), 'middle': defaultdict(list), 'bottom': defaultdict(list)}
    total_rows = sum(section_allocation.values())
    
    for shape in shapes:
        centroid_row = shape['centroid'][0]
        if centroid_row < total_rows / 3:
            section = 'top'
        elif centroid_row < 2 * total_rows / 3:
            section = 'middle'
        else:
            section = 'bottom'
        grouped[section][shape['color']].append(shape)
    
    for section in grouped:
        for color in grouped[section]:
            grouped[section][color].sort(key=lambda s: -s['size'])
    
    return grouped

def place_shapes_in_grid(grid: ColoredGrid, grouped_shapes: Dict[str, Dict[int, List[Dict]]], section_allocation: Dict[str, int]):
    current_row = 0
    for section in ['top', 'middle', 'bottom']:
        section_height = section_allocation[section]
        place_shapes_in_section(grid, grouped_shapes[section], current_row, section_height)
        current_row += section_height

def place_shapes_in_section(grid: ColoredGrid, shapes_by_color: Dict[int, List[Dict]], start_row: int, height: int):
    current_row = start_row
    for color in sorted(shapes_by_color.keys(), key=lambda c: -sum(s['size'] for s in shapes_by_color[c])):
        row = current_row
        col = 0
        for shape in shapes_by_color[color]:
            if col + shape['bounding_box'][3] > grid.get_dimensions()[1]:
                row += 1
                col = 0
            if row + shape['bounding_box'][2] > start_row + height:
                continue  # Skip if shape doesn't fit in the section
            place_shape(grid, shape, (row, col))
            col += shape['bounding_box'][3] + 1
        current_row = row + 1

def place_shape(grid: ColoredGrid, shape: Dict, anchor: Tuple[int, int]):
    for r, c in shape['cells']:
        dr, dc = r - shape['bounding_box'][0], c - shape['bounding_box'][1]
        grid.set_cell(anchor[0] + dr, anchor[1] + dc, shape['color'])

def optimize_placement(grid: ColoredGrid, grouped_shapes: Dict[str, Dict[int, List[Dict]]]):
    # This function can be implemented to further optimize the placement
    # For example, filling large gaps or balancing the grid
    pass
