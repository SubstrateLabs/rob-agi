from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_9f27f097(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying a source region with diverse colors,
    finding a target region (usually single-color), and copying the source to the target
    with an appropriate transformation (180-degree rotation, vertical flip, or no change).
    
    1. Identify the border color
    2. Find the source region (most color-diverse non-border region)
    3. Find the target region (single-color region, or largest contiguous area)
    4. Determine and apply the appropriate transformation
    5. Copy the transformed source region to the target region
    """
    # Step 1: Identify border color
    border_color = identify_border_color(input_grid)
    
    # Step 2: Find source region
    source_region = find_most_diverse_region(input_grid, border_color)
    
    # Step 3: Find target region
    target_region = find_target_region(input_grid, border_color, len(source_region))
    
    # Step 4 & 5: Determine transformation, apply it, and copy
    output_grid = input_grid.deep_copy()
    if source_region and target_region:
        source_bounds = get_region_bounds(source_region)
        target_bounds = get_region_bounds(target_region)
        transformation = determine_transformation(input_grid, source_region, target_region)
        
        for sy, sx in source_region:
            ty, tx = transform(sy, sx, source_bounds, target_bounds, transformation)
            output_grid.values[ty][tx] = input_grid.values[sy][sx]
    
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
