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
    regions = input_grid.find_connected_regions(border_color)
    source_region = max(regions, key=lambda r: len(set(input_grid.values[y][x] for y, x in r)))
    
    # Step 3: Find target region
    target_region = next(r for r in regions if len(set(input_grid.values[y][x] for y, x in r)) == 1 and len(r) == len(source_region))
    
    # Step 4 & 5: Apply transformation and copy
    output_grid = input_grid.deep_copy()
    source_center = get_region_center(source_region)
    target_center = get_region_center(target_region)
    
    for sy, sx in source_region:
        ty, tx = transform(sy, sx, source_center, target_center)
        if 0 <= ty < len(output_grid.values) and 0 <= tx < len(output_grid.values[0]):
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

def get_region_center(region: List[Tuple[int, int]]) -> Tuple[float, float]:
    y_coords, x_coords = zip(*region)
    return (sum(y_coords) / len(y_coords), sum(x_coords) / len(x_coords))

def transform(y: int, x: int, source_center: Tuple[float, float], target_center: Tuple[float, float]) -> Tuple[int, int]:
    dy = target_center[0] - source_center[0]
    dx = target_center[1] - source_center[1]
    new_y = int(target_center[0] - (y - source_center[0]))
    new_x = int(target_center[1] - (x - source_center[1]))
    return new_y, new_x
