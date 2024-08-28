from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_9f27f097(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying a source region with diverse colors,
    finding a target region, and applying a transformation (usually a 180-degree rotation).
    
    1. Identify the border color
    2. Find the source region (area with the most diverse colors)
    3. Find the target region (preferably black, single-color, or largest non-border region)
    4. Determine the transformation type (rotation, reflection, or translation)
    5. Apply the transformation from source to target
    6. Handle size differences between source and target regions
    7. If no suitable target region is found, create one in the opposite corner
    8. Ensure the border and grid size remain unchanged
    """
    # Step 1: Identify border color
    border_color = identify_border_color(input_grid)
    
    # Step 2: Find source region
    source_region = find_most_diverse_region(input_grid, border_color)
    
    # Step 3 & 4: Find target region and determine size
    source_size = len(source_region)
    target_region = find_target_region(input_grid, border_color, source_size)
    
    # If no suitable target region is found, create one in the opposite corner
    if not target_region:
        source_bounds = get_region_bounds(source_region)
        target_region = create_opposite_corner_region(input_grid, source_bounds, border_color)
    
    # Step 5 & 6: Apply rotation and handle expansion
    output_grid = apply_transformation(input_grid, source_region, target_region)
    
    return output_grid

def apply_transformation(input_grid: ColoredGrid, source_region: List[Tuple[int, int]], target_region: List[Tuple[int, int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    source_bounds = get_region_bounds(source_region)
    target_bounds = get_region_bounds(target_region)
    
    transformation = determine_transformation(source_bounds, target_bounds)
    
    for sy, sx in source_region:
        ty, tx = transform(sy, sx, source_bounds, target_bounds, transformation)
        if 0 <= ty < input_grid.num_rows and 0 <= tx < input_grid.num_cols:
            output_grid.values[ty][tx] = input_grid.values[sy][sx]
    
    return output_grid

def determine_transformation(source_bounds: Tuple[int, int, int, int], target_bounds: Tuple[int, int, int, int]) -> str:
    sy_min, sx_min, sy_max, sx_max = source_bounds
    ty_min, tx_min, ty_max, tx_max = target_bounds
    
    if (sy_min + sy_max) // 2 < (ty_min + ty_max) // 2 and (sx_min + sx_max) // 2 < (tx_min + tx_max) // 2:
        return "rotate_180"
    elif sy_min == ty_min and sx_min != tx_min:
        return "horizontal_flip"
    elif sy_min != ty_min and sx_min == tx_min:
        return "vertical_flip"
    else:
        return "translate"

def find_most_diverse_region(grid: ColoredGrid, border_color: int) -> List[Tuple[int, int]]:
    regions = grid.find_connected_regions(lambda x: x != border_color)
    if not regions:
        return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] != border_color]
    return max(regions, key=lambda r: len(set(grid.values[y][x] for y, x in r)))

def find_target_region(grid: ColoredGrid, border_color: int, size: int) -> Optional[List[Tuple[int, int]]]:
    regions = grid.find_connected_regions(lambda x: x != border_color)
    
    # First, look for black regions
    black_regions = [r for r in regions if all(grid.values[y][x] == 0 for y, x in r)]
    if black_regions:
        target = max(black_regions, key=len)
        if len(target) >= size:
            return target[:size]
        return expand_region(grid, target, size, border_color)
    
    # If no black regions, look for single color regions
    single_color_regions = [r for r in regions if len(set(grid.values[y][x] for y, x in r)) == 1]
    if single_color_regions:
        target = max(single_color_regions, key=len)
        if len(target) >= size:
            return target[:size]
        return expand_region(grid, target, size, border_color)
    
    # If no suitable region found, return None
    return None

def expand_region(grid: ColoredGrid, region: List[Tuple[int, int]], target_size: int, border_color: int) -> List[Tuple[int, int]]:
    expanded = set(region)
    queue = list(region)
    while len(expanded) < target_size and queue:
        y, x = queue.pop(0)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols and (ny, nx) not in expanded and grid.values[ny][nx] != border_color:
                expanded.add((ny, nx))
                queue.append((ny, nx))
                if len(expanded) == target_size:
                    break
    return list(expanded)

def get_region_bounds(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    y_coords, x_coords = zip(*region)
    return min(y_coords), min(x_coords), max(y_coords), max(x_coords)

def transform(y: int, x: int, source_bounds: Tuple[int, int, int, int], target_bounds: Tuple[int, int, int, int], transformation: str) -> Tuple[int, int]:
    sy_min, sx_min, sy_max, sx_max = source_bounds
    ty_min, tx_min, ty_max, tx_max = target_bounds
    
    if transformation == "rotate_180":
        new_y = sy_max - (y - sy_min)
        new_x = sx_max - (x - sx_min)
    elif transformation == "horizontal_flip":
        new_y = y - sy_min
        new_x = sx_max - (x - sx_min)
    elif transformation == "vertical_flip":
        new_y = sy_max - (y - sy_min)
        new_x = x - sx_min
    else:  # translate
        new_y = y - sy_min
        new_x = x - sx_min
    
    ty = ty_min + new_y
    tx = tx_min + new_x
    
    return ty, tx

def identify_border_color(grid: ColoredGrid) -> int:
    rows, cols = grid.get_dimensions()
    border_cells = (
        [(0, c) for c in range(cols)] +
        [(rows-1, c) for c in range(cols)] +
        [(r, 0) for r in range(1, rows-1)] +
        [(r, cols-1) for r in range(1, rows-1)]
    )
    return max(set(grid.values[r][c] for r, c in border_cells), key=lambda color: sum(grid.values[r][c] == color for r, c in border_cells))

def find_most_diverse_corner(grid: ColoredGrid, border_color: int) -> str:
    corners = {
        'top_left': [(r, c) for r in range(1, 6) for c in range(1, 6)],
        'top_right': [(r, c) for r in range(1, 6) for c in range(6, 11)],
        'bottom_left': [(r, c) for r in range(6, 11) for c in range(1, 6)],
        'bottom_right': [(r, c) for r in range(6, 11) for c in range(6, 11)]
    }
    
    max_diversity = 0
    most_diverse_corner = ''
    
    for corner, cells in corners.items():
        unique_colors = len(set(grid.values[r][c] for r, c in cells if grid.values[r][c] != border_color))
        if unique_colors > max_diversity:
            max_diversity = unique_colors
            most_diverse_corner = corner
    
    return most_diverse_corner

def get_opposite_corner(corner: str) -> str:
    opposites = {
        'top_left': 'bottom_right',
        'top_right': 'bottom_left',
        'bottom_left': 'top_right',
        'bottom_right': 'top_left'
    }
    return opposites[corner]

def copy_with_rotation(input_grid: ColoredGrid, output_grid: ColoredGrid, source_corner: str, target_corner: str):
    corners = {
        'top_left': (1, 1),
        'top_right': (1, 6),
        'bottom_left': (6, 1),
        'bottom_right': (6, 6)
    }
    
    sr, sc = corners[source_corner]
    tr, tc = corners[target_corner]
    
    for i in range(5):
        for j in range(5):
            output_grid.values[tr+4-i][tc+4-j] = input_grid.values[sr+i][sc+j]

def identify_border_color(grid: ColoredGrid) -> int:
    rows, cols = grid.get_dimensions()
    border_cells = (
        [(0, c) for c in range(cols)] +
        [(rows-1, c) for c in range(cols)] +
        [(r, 0) for r in range(1, rows-1)] +
        [(r, cols-1) for r in range(1, rows-1)]
    )
    return max(set(grid.values[r][c] for r, c in border_cells), key=lambda color: sum(grid.values[r][c] == color for r, c in border_cells))

def find_most_diverse_region(grid: ColoredGrid, border_color: int) -> List[Tuple[int, int]]:
    regions = grid.find_connected_regions(lambda x: x != border_color)
    if not regions:
        return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] != border_color]
    return max(regions, key=lambda r: len(set(grid.values[y][x] for y, x in r)))

def find_target_region(grid: ColoredGrid, border_color: int, size: int) -> Optional[List[Tuple[int, int]]]:
    regions = grid.find_connected_regions(lambda x: x != border_color)
    single_color_regions = [r for r in regions if len(set(grid.values[y][x] for y, x in r)) == 1]
    
    if single_color_regions:
        target = max(single_color_regions, key=len)
        if len(target) < size:
            return expand_region(grid, target, size, border_color)
        return target[:size]
    
    return max(regions, key=len)[:size] if regions else None

def expand_region(grid: ColoredGrid, region: List[Tuple[int, int]], target_size: int, border_color: int) -> List[Tuple[int, int]]:
    expanded = set(region)
    queue = list(region)
    while len(expanded) < target_size and queue:
        y, x = queue.pop(0)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols and (ny, nx) not in expanded and grid.values[ny][nx] != border_color:
                expanded.add((ny, nx))
                queue.append((ny, nx))
                if len(expanded) == target_size:
                    break
    return list(expanded)

def get_region_bounds(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    y_coords, x_coords = zip(*region)
    return min(y_coords), min(x_coords), max(y_coords), max(x_coords)

def determine_transformation(grid: ColoredGrid, source: List[Tuple[int, int]], target: List[Tuple[int, int]]) -> str:
    source_colors = [grid.values[y][x] for y, x in source]
    target_colors = [grid.values[y][x] for y, x in target]
    
    if source_colors == target_colors[::-1]:
        return "rotate_180"
    elif source_colors == target_colors[::-1][:]:
        return "vertical_flip"
    else:
        return "no_change"

def transform(y: int, x: int, source_bounds: Tuple[int, int, int, int], target_bounds: Tuple[int, int, int, int], transformation: str) -> Tuple[int, int]:
    sy_min, sx_min, sy_max, sx_max = source_bounds
    ty_min, tx_min, ty_max, tx_max = target_bounds
    
    if transformation == "rotate_180":
        new_y = sy_max - (y - sy_min)
        new_x = sx_max - (x - sx_min)
    elif transformation == "vertical_flip":
        new_y = sy_max - (y - sy_min)
        new_x = x - sx_min
    else:  # no_change
        new_y = y - sy_min
        new_x = x - sx_min
    
    ty = ty_min + new_y
    tx = tx_min + new_x
    
    return ty, tx
def create_opposite_corner_region(grid: ColoredGrid, source_bounds: Tuple[int, int, int, int], border_color: int) -> List[Tuple[int, int]]:
    sy_min, sx_min, sy_max, sx_max = source_bounds
    rows, cols = grid.get_dimensions()
    
    # Determine if the source is in the top-left or bottom-right quadrant
    if sy_min < rows // 2 and sx_min < cols // 2:
        # Source is in top-left, create target in bottom-right
        ty_min, tx_min = rows - (sy_max - sy_min) - 1, cols - (sx_max - sx_min) - 1
    else:
        # Source is in bottom-right, create target in top-left
        ty_min, tx_min = 1, 1
    
    ty_max, tx_max = ty_min + (sy_max - sy_min), tx_min + (sx_max - sx_min)
    
    return [(y, x) for y in range(ty_min, ty_max + 1) for x in range(tx_min, tx_max + 1) if grid.values[y][x] != border_color]
