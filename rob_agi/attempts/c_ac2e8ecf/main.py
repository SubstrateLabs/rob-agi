from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_ac2e8ecf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging shapes based on their properties and positions.
    
    The solution follows these steps:
    1. Analyze the input grid to identify all shapes and their properties.
    2. Assign priority scores to shapes based on size, regularity, and original position.
    3. Create a new grid and define anchor points for shape placement.
    4. Place shapes in the new grid according to their priority and available space.
    5. Handle any remaining unplaced shapes.
    6. Fill empty spaces with black (0).

    This approach aims to create a balanced and organized output grid while preserving
    the original shapes and their relative positions.
    """
    shapes = analyze_shapes(input_grid)
    shapes.sort(key=lambda s: calculate_priority(s, input_grid.get_dimensions()), reverse=True)
    
    new_grid = ColoredGrid(values=[[0 for _ in range(input_grid.get_dimensions()[1])] 
                                   for _ in range(input_grid.get_dimensions()[0])])
    
    anchor_points = generate_anchor_points(input_grid.get_dimensions())
    
    for shape in shapes:
        placed = False
        for anchor in anchor_points:
            if place_shape(new_grid, shape, anchor):
                placed = True
                break
        if not placed:
            place_remaining_shape(new_grid, shape)
    
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
                'original_position': get_center(region),
                'cells': region
            }
            shape['regularity'] = shape['size'] / (shape['bounding_box'][2] * shape['bounding_box'][3])
            shapes.append(shape)
    return shapes

def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    return (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)

def get_center(region: List[Tuple[int, int]]) -> Tuple[float, float]:
    return (sum(r for r, _ in region) / len(region), sum(c for _, c in region) / len(region))

def calculate_priority(shape: Dict, grid_size: Tuple[int, int]) -> float:
    size_score = shape['size'] / (grid_size[0] * grid_size[1])
    regularity_score = shape['regularity']
    edge_score = calculate_edge_score(shape['original_position'], grid_size)
    return size_score + regularity_score + edge_score

def calculate_edge_score(position: Tuple[float, float], grid_size: Tuple[int, int]) -> float:
    row, col = position
    return min(row, col, grid_size[0] - 1 - row, grid_size[1] - 1 - col) / max(grid_size)

def generate_anchor_points(grid_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid_size
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    edges = [(0, c) for c in range(1, cols-1)] + [(rows-1, c) for c in range(1, cols-1)] + \
            [(r, 0) for r in range(1, rows-1)] + [(r, cols-1) for r in range(1, rows-1)]
    interior = [(r, c) for r in range(1, rows-1) for c in range(1, cols-1)]
    return corners + edges + interior

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

def place_remaining_shape(grid: ColoredGrid, shape: Dict):
    rows, cols = grid.get_dimensions()
    for r in range(rows - shape['bounding_box'][2] + 1):
        for c in range(cols - shape['bounding_box'][3] + 1):
            if place_shape(grid, shape, (r, c)):
                return
