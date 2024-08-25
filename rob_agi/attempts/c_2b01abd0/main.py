from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_2b01abd0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by following these steps:
    1. Identifies the blue line dividing the grid
    2. Determines the source and target sides
    3. Creates a mirrored copy of the source side on the target side
    4. Identifies connected regions in the source side
    5. For each region, swaps the main color with the inner color (if exists)
       in both the original and mirrored positions
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid
    """
    blue_line = find_blue_line(input_grid)
    source_side, target_side = determine_sides(input_grid, blue_line)
    new_grid = create_mirrored_grid(input_grid, blue_line, source_side, target_side)
    regions = find_connected_regions(input_grid, source_side)
    for region in regions:
        main_color, inner_color = identify_colors(region)
        if inner_color:
            swap_colors(new_grid, region, main_color, inner_color)
            mirrored_region = mirror_region(region, blue_line)
            swap_colors(new_grid, mirrored_region, main_color, inner_color)
    return new_grid

def find_blue_line(grid: ColoredGrid) -> Tuple[str, int]:
    rows, cols = grid.get_dimensions()
    for i in range(rows):
        if all(cell == 1 for cell in grid.values[i]):
            return 'horizontal', i
    for j in range(cols):
        if all(grid.values[r][j] == 1 for r in range(rows)):
            return 'vertical', j
    raise ValueError("No blue line found")

def determine_sides(grid: ColoredGrid, blue_line: Tuple[str, int]) -> Tuple[str, str]:
    orientation, position = blue_line
    rows, cols = grid.get_dimensions()
    if orientation == 'horizontal':
        top_count = sum(sum(row) != 0 for row in grid.values[:position])
        bottom_count = sum(sum(row) != 0 for row in grid.values[position+1:])
        return ('top', 'bottom') if top_count > bottom_count else ('bottom', 'top')
    else:
        left_count = sum(sum(row[:position]) != 0 for row in grid.values)
        right_count = sum(sum(row[position+1:]) != 0 for row in grid.values)
        return ('left', 'right') if left_count > right_count else ('right', 'left')

def create_mirrored_grid(grid: ColoredGrid, blue_line: Tuple[str, int], source_side: str, target_side: str) -> ColoredGrid:
    new_grid = grid.deep_copy()
    orientation, position = blue_line
    rows, cols = grid.get_dimensions()
    if orientation == 'horizontal':
        source_range = range(position) if source_side == 'top' else range(position + 1, rows)
        for r in source_range:
            mirrored_r = position - (r - position) if source_side == 'top' else position + (position - r)
            new_grid.values[mirrored_r] = grid.values[r].copy()
    else:
        source_range = range(position) if source_side == 'left' else range(position + 1, cols)
        for c in source_range:
            mirrored_c = position - (c - position) if source_side == 'left' else position + (position - c)
            for r in range(rows):
                new_grid.values[r][mirrored_c] = grid.values[r][c]
    return new_grid

def find_connected_regions(grid: ColoredGrid, side: str) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    if side in ['top', 'bottom']:
        start, end = (0, rows // 2) if side == 'top' else (rows // 2, rows)
        return [region for color in range(10) for region in grid.find_connected_regions(color) if all(start <= r < end for r, _ in region)]
    else:
        start, end = (0, cols // 2) if side == 'left' else (cols // 2, cols)
        return [region for color in range(10) for region in grid.find_connected_regions(color) if all(start <= c < end for _, c in region)]

def identify_colors(region: List[Tuple[int, int]]) -> Tuple[int, int]:
    colors = [grid.values[r][c] for r, c in region]
    main_color = max(set(colors), key=colors.count)
    inner_colors = set(colors) - {main_color, 0}
    return main_color, inner_colors.pop() if inner_colors else None

def swap_colors(grid: ColoredGrid, region: List[Tuple[int, int]], color1: int, color2: int):
    for r, c in region:
        if grid.values[r][c] == color1:
            grid.values[r][c] = color2
        elif grid.values[r][c] == color2:
            grid.values[r][c] = color1

def mirror_region(region: List[Tuple[int, int]], blue_line: Tuple[str, int]) -> List[Tuple[int, int]]:
    orientation, position = blue_line
    if orientation == 'horizontal':
        return [(2 * position - r, c) for r, c in region]
    else:
        return [(r, 2 * position - c) for r, c in region]
