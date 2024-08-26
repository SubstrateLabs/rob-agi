from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_9f27f097(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying a source region with diverse colors,
    finding a target region (usually single-color), and copying the source to the target
    with a 180-degree rotation.
    
    1. Identify the border color
    2. Find the source region (5x5 corner with the most diverse colors)
    3. Find the target region (opposite corner to the source)
    4. Apply 180-degree rotation from source to target
    5. Copy the transformed source region to the target region
    """
    # Step 1: Identify border color
    border_color = identify_border_color(input_grid)
    
    # Step 2: Find source region
    source_corner = find_most_diverse_corner(input_grid, border_color)
    
    # Step 3: Find target region
    target_corner = get_opposite_corner(source_corner)
    
    # Step 4 & 5: Apply 180-degree rotation and copy
    output_grid = input_grid.deep_copy()
    copy_with_rotation(input_grid, output_grid, source_corner, target_corner)
    
    return output_grid

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
