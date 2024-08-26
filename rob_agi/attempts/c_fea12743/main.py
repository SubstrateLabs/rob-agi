from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
from collections import namedtuple

Region = namedtuple('Region', ['id', 'color', 'cells', 'centroid', 'quadrant'])

def solve_fea12743(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the fea12743 challenge by dividing the grid into quadrants,
    identifying shapes in each quadrant, analyzing quadrant connectivity,
    and applying color transformations based on the identified pattern:
    - The source quadrant (usually bottom-right or the most connected) remains red (2)
    - The next quadrant clockwise becomes green (3)
    - The remaining two quadrants become sky blue (8)
    - Black cells (0) are preserved
    - Special case: bottom-right quadrant is never sky blue
    """
    # Step 1-2: Parse the input grid and divide into quadrants
    rows, cols = input_grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2

    # Step 3: Identify shapes in each quadrant
    regions = find_regions(input_grid, mid_row, mid_col)

    # Step 4-5: Analyze quadrant connectivity and determine the source quadrant
    quadrant_connections = analyze_quadrant_connectivity(regions)
    source_quadrant = determine_source_quadrant(quadrant_connections)

    # Step 6-7: Establish color transformation order and apply it
    color_order = get_color_transformation_order(source_quadrant)
    new_grid = apply_color_transformation(input_grid, regions, color_order)

    # Step 8: Handle special case (bottom-right never sky blue)
    handle_bottom_right_special_case(new_grid, mid_row, mid_col)

    return new_grid

def find_regions(grid: ColoredGrid, mid_row: int, mid_col: int) -> List[Region]:
    regions = []
    visited = set()
    region_id = 0
    for x in range(grid.get_dimensions()[0]):
        for y in range(grid.get_dimensions()[1]):
            if (x, y) not in visited and grid.values[x][y] != 0:
                cells = flood_fill(grid, x, y, grid.values[x][y])
                centroid = calculate_centroid(cells)
                quadrant = get_quadrant(centroid, mid_row, mid_col)
                regions.append(Region(region_id, grid.values[x][y], cells, centroid, quadrant))
                visited.update(cells)
                region_id += 1
    return regions

def flood_fill(grid: ColoredGrid, x: int, y: int, color: int) -> Set[Tuple[int, int]]:
    cells = set()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in cells or grid.values[cx][cy] != color:
            continue
        cells.add((cx, cy))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid.get_dimensions()[0] and 0 <= ny < grid.get_dimensions()[1]:
                stack.append((nx, ny))
    return cells

def calculate_centroid(cells: Set[Tuple[int, int]]) -> Tuple[float, float]:
    return sum(x for x, _ in cells) / len(cells), sum(y for _, y in cells) / len(cells)

def get_quadrant(centroid: Tuple[float, float], mid_row: int, mid_col: int) -> str:
    x, y = centroid
    if x < mid_row:
        return 'top-left' if y < mid_col else 'top-right'
    else:
        return 'bottom-left' if y < mid_col else 'bottom-right'

def analyze_quadrant_connectivity(regions: List[Region]) -> Dict[str, Set[str]]:
    connections = {'top-left': set(), 'top-right': set(), 'bottom-left': set(), 'bottom-right': set()}
    for region in regions:
        for other_region in regions:
            if region != other_region and are_adjacent(region.cells, other_region.cells):
                connections[region.quadrant].add(other_region.quadrant)
    return connections

def are_adjacent(cells1: Set[Tuple[int, int]], cells2: Set[Tuple[int, int]]) -> bool:
    return any(abs(x1 - x2) + abs(y1 - y2) == 1 for x1, y1 in cells1 for x2, y2 in cells2)

def determine_source_quadrant(connections: Dict[str, Set[str]]) -> str:
    if len(connections['bottom-right']) > 1:
        return 'bottom-right'
    return max(connections, key=lambda q: len(connections[q]))

def get_color_transformation_order(source_quadrant: str) -> List[str]:
    order = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
    start_index = order.index(source_quadrant)
    return order[start_index:] + order[:start_index]

def apply_color_transformation(grid: ColoredGrid, regions: List[Region], color_order: List[str]) -> ColoredGrid:
    new_grid = ColoredGrid(values=[[0 for _ in range(grid.get_dimensions()[1])] 
                                   for _ in range(grid.get_dimensions()[0])])
    color_map = {color_order[0]: 2, color_order[1]: 3, color_order[2]: 8, color_order[3]: 8}
    
    for region in regions:
        new_color = color_map[region.quadrant]
        for x, y in region.cells:
            new_grid.values[x][y] = new_color
    
    # Preserve black cells
    for x in range(grid.get_dimensions()[0]):
        for y in range(grid.get_dimensions()[1]):
            if grid.values[x][y] == 0:
                new_grid.values[x][y] = 0
    
    return new_grid

def handle_bottom_right_special_case(grid: ColoredGrid, mid_row: int, mid_col: int):
    bottom_right_color = grid.values[mid_row][mid_col]
    if bottom_right_color == 8:  # If bottom-right is sky blue, swap with the next counterclockwise quadrant
        for x in range(mid_row, grid.get_dimensions()[0]):
            for y in range(mid_col, grid.get_dimensions()[1]):
                if grid.values[x][y] != 0:
                    grid.values[x][y] = 2  # Change to red
        for x in range(mid_row):
            for y in range(mid_col, grid.get_dimensions()[1]):
                if grid.values[x][y] != 0:
                    grid.values[x][y] = 8  # Change to sky blue
