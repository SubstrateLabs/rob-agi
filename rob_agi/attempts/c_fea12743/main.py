from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
from collections import namedtuple

Region = namedtuple('Region', ['id', 'color', 'cells', 'centroid', 'quadrant'])

def solve_fea12743(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the fea12743 challenge by:
    1. Dividing the grid into quadrants
    2. Analyzing each quadrant for size and complexity of red shapes
    3. Ranking quadrants based on a scoring system
    4. Applying color transformations:
       - Highest-ranked quadrant remains red (2)
       - Second-highest becomes green (3), unless it's bottom-right
       - Remaining quadrants become sky blue (8)
       - Bottom-right is never sky blue, becomes green (3) if not highest-ranked
    5. Preserving black cells (0)
    """
    # Step 1: Divide the grid into quadrants
    rows, cols = input_grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2

    # Step 2-3: Analyze quadrants and rank them
    quadrant_scores = analyze_and_score_quadrants(input_grid, mid_row, mid_col)
    ranked_quadrants = rank_quadrants(quadrant_scores)

    # Step 4-5: Apply color transformation
    new_grid = apply_color_transformation(input_grid, ranked_quadrants, mid_row, mid_col)

    return new_grid

def analyze_and_score_quadrants(grid: ColoredGrid, mid_row: int, mid_col: int) -> Dict[str, float]:
    quadrant_scores = {'top-left': 0, 'top-right': 0, 'bottom-left': 0, 'bottom-right': 0}
    for quadrant in quadrant_scores:
        regions = find_regions(grid, quadrant, mid_row, mid_col)
        size = sum(len(region) for region in regions)
        complexity = sum(calculate_complexity(region) for region in regions)
        quadrant_scores[quadrant] = size * 0.7 + complexity * 0.3
    return quadrant_scores

def find_regions(grid: ColoredGrid, quadrant: str, mid_row: int, mid_col: int) -> List[Set[Tuple[int, int]]]:
    regions = []
    visited = set()
    row_range = range(mid_row) if 'top' in quadrant else range(mid_row, grid.num_rows)
    col_range = range(mid_col) if 'left' in quadrant else range(mid_col, grid.num_cols)
    
    for r in row_range:
        for c in col_range:
            if grid.values[r][c] == 2 and (r, c) not in visited:
                region = flood_fill(grid, r, c, 2)
                regions.append(region)
                visited.update(region)
    return regions

def calculate_complexity(region: Set[Tuple[int, int]]) -> int:
    return sum(1 for r, c in region if any((r+dr, c+dc) not in region for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]))

def rank_quadrants(quadrant_scores: Dict[str, float]) -> List[str]:
    return sorted(quadrant_scores, key=quadrant_scores.get, reverse=True)

def apply_color_transformation(grid: ColoredGrid, ranked_quadrants: List[str], mid_row: int, mid_col: int) -> ColoredGrid:
    new_grid = grid.deep_copy()
    color_map = {ranked_quadrants[0]: 2}
    
    if ranked_quadrants[0] != 'bottom-right':
        color_map['bottom-right'] = 3
        color_map[ranked_quadrants[1]] = 3
    else:
        color_map[ranked_quadrants[1]] = 3
    
    for quadrant in ranked_quadrants[2:]:
        if quadrant != 'bottom-right':
            color_map[quadrant] = 8
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:
                quadrant = get_quadrant(r, c, mid_row, mid_col)
                new_grid.values[r][c] = color_map[quadrant]
    
    return new_grid

def get_quadrant(r: int, c: int, mid_row: int, mid_col: int) -> str:
    if r < mid_row:
        return 'top-left' if c < mid_col else 'top-right'
    else:
        return 'bottom-left' if c < mid_col else 'bottom-right'

def flood_fill(grid: ColoredGrid, r: int, c: int, color: int) -> Set[Tuple[int, int]]:
    region = set()
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if (r, c) not in region and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            region.add((r, c))
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    return region

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
