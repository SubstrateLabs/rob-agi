from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_9f27f097(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying a source region with diverse colors,
    finding a target region (usually single-color or empty), and copying the source to the target
    with a 180-degree rotation.
    
    1. Identify the border color
    2. Find the source region (corner with the most diverse colors)
    3. Find the target region (opposite corner to the source)
    4. Determine the size of the source region
    5. Apply 180-degree rotation from source to target
    6. Handle target region expansion if necessary
    """
    # Step 1: Identify border color
    border_color = identify_border_color(input_grid)
    
    # Step 2 & 3: Find source and target regions
    source_corner, target_corner = find_source_and_target_corners(input_grid, border_color)
    
    # Step 4: Determine size of source region
    source_size = determine_source_size(input_grid, source_corner, border_color)
    
    # Step 5 & 6: Apply rotation and handle expansion
    output_grid = apply_transformation(input_grid, source_corner, target_corner, source_size, border_color)
    
    return output_grid

def find_source_and_target_corners(grid: ColoredGrid, border_color: int) -> Tuple[str, str]:
    corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    corner_diversity = {corner: count_unique_colors(grid, corner, border_color) for corner in corners}
    source_corner = max(corner_diversity, key=corner_diversity.get)
    target_corner = get_opposite_corner(source_corner)
    return source_corner, target_corner

def count_unique_colors(grid: ColoredGrid, corner: str, border_color: int) -> int:
    start_row, start_col = get_corner_coordinates(corner, grid.num_rows, grid.num_cols)
    unique_colors = set()
    for r in range(start_row, start_row + 5):
        for c in range(start_col, start_col + 5):
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                color = grid.values[r][c]
                if color != border_color:
                    unique_colors.add(color)
    return len(unique_colors)

def determine_source_size(grid: ColoredGrid, corner: str, border_color: int) -> int:
    start_row, start_col = get_corner_coordinates(corner, grid.num_rows, grid.num_cols)
    size = 0
    while True:
        if (start_row + size >= grid.num_rows or start_col + size >= grid.num_cols or
            grid.values[start_row + size][start_col + size] == border_color):
            break
        size += 1
    return size

def apply_transformation(grid: ColoredGrid, source_corner: str, target_corner: str, size: int, border_color: int) -> ColoredGrid:
    output_grid = grid.deep_copy()
    source_row, source_col = get_corner_coordinates(source_corner, grid.num_rows, grid.num_cols)
    target_row, target_col = get_corner_coordinates(target_corner, grid.num_rows, grid.num_cols)
    
    for r in range(size):
        for c in range(size):
            source_value = grid.values[source_row + r][source_col + c]
            new_r = target_row + (size - 1 - r)
            new_c = target_col + (size - 1 - c)
            if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
                output_grid.values[new_r][new_c] = source_value
    
    return output_grid

def get_corner_coordinates(corner: str, num_rows: int, num_cols: int) -> Tuple[int, int]:
    if corner == 'top_left':
        return 0, 0
    elif corner == 'top_right':
        return 0, num_cols - 1
    elif corner == 'bottom_left':
        return num_rows - 1, 0
    else:  # bottom_right
        return num_rows - 1, num_cols - 1

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
