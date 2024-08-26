from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9f27f097(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying a source region with diverse colors,
    finding a target region of the same size (usually black), and copying the source to the target
    with a 180-degree rotation (mirroring both horizontally and vertically).
    
    1. Identify the border color
    2. Find the source region (most color-diverse non-border region)
    3. Find the target region (single-color region of the same size as source)
    4. Apply the transformation (translate and rotate 180 degrees)
    5. Copy the transformed source region to the target region
    """
    # Step 1: Identify border color
    border_color = identify_border_color(input_grid)
    
    # Step 2: Find source region
    source_region = find_most_diverse_region(input_grid, border_color)
    
    # Step 3: Find target region
    target_region = find_target_region(input_grid, border_color, len(source_region))
    
    # Step 4 & 5: Apply transformation and copy
    output_grid = input_grid.deep_copy()
    source_bounds = get_region_bounds(source_region)
    target_bounds = get_region_bounds(target_region)
    
    for sy, sx in source_region:
        ty, tx = transform(sy, sx, source_bounds, target_bounds)
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
    return max(regions, key=lambda r: len(set(grid.values[y][x] for y, x in r)))

def find_target_region(grid: ColoredGrid, border_color: int, size: int) -> List[Tuple[int, int]]:
    regions = grid.find_connected_regions(lambda x: x != border_color)
    return next(r for r in regions if len(set(grid.values[y][x] for y, x in r)) == 1 and len(r) == size)

def get_region_bounds(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    y_coords, x_coords = zip(*region)
    return min(y_coords), min(x_coords), max(y_coords), max(x_coords)

def transform(y: int, x: int, source_bounds: Tuple[int, int, int, int], target_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
    sy_min, sx_min, sy_max, sx_max = source_bounds
    ty_min, tx_min, ty_max, tx_max = target_bounds
    
    # Flip coordinates within the source region
    new_y = sy_max - (y - sy_min)
    new_x = sx_max - (x - sx_min)
    
    # Map to target region
    ty = ty_min + (new_y - sy_min)
    tx = tx_min + (new_x - sx_min)
    
    return ty, tx
