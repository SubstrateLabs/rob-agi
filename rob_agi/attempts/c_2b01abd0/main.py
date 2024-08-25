from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_2b01abd0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by following these steps:
    1. Identifies the blue line dividing the grid
    2. Determines the source and target sides
    3. Identifies the two most common non-black, non-blue colors in the source side
    4. Creates a mirrored copy of the source side on the target side, swapping the two most common colors
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid
    """
    blue_line = find_blue_line(input_grid)
    source_side, target_side = determine_sides(input_grid, blue_line)
    color1, color2 = identify_common_colors(input_grid, blue_line, source_side)
    new_grid = create_mirrored_grid(input_grid, blue_line, source_side, target_side, color1, color2)
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

def identify_common_colors(grid: ColoredGrid, blue_line: Tuple[str, int], source_side: str) -> Tuple[int, int]:
    color_counts = {}
    orientation, position = blue_line
    rows, cols = grid.get_dimensions()
    
    if orientation == 'horizontal':
        range_to_check = range(position) if source_side == 'top' else range(position + 1, rows)
        for r in range_to_check:
            for c in range(cols):
                color = grid.values[r][c]
                if color not in [0, 1]:  # Exclude black and blue
                    color_counts[color] = color_counts.get(color, 0) + 1
    else:
        range_to_check = range(position) if source_side == 'left' else range(position + 1, cols)
        for r in range(rows):
            for c in range_to_check:
                color = grid.values[r][c]
                if color not in [0, 1]:  # Exclude black and blue
                    color_counts[color] = color_counts.get(color, 0) + 1
    
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_colors[0][0], sorted_colors[1][0] if len(sorted_colors) > 1 else sorted_colors[0][0]

def create_mirrored_grid(grid: ColoredGrid, blue_line: Tuple[str, int], source_side: str, target_side: str, color1: int, color2: int) -> ColoredGrid:
    new_grid = grid.deep_copy()
    orientation, position = blue_line
    rows, cols = grid.get_dimensions()
    
    if orientation == 'horizontal':
        source_range = range(position) if source_side == 'top' else range(position + 1, rows)
        for r in source_range:
            mirrored_r = position - (r - position) if source_side == 'top' else position + (position - r)
            for c in range(cols):
                original_color = grid.values[r][c]
                if original_color == color1:
                    new_grid.values[mirrored_r][c] = color2
                elif original_color == color2:
                    new_grid.values[mirrored_r][c] = color1
                else:
                    new_grid.values[mirrored_r][c] = original_color
    else:
        source_range = range(position) if source_side == 'left' else range(position + 1, cols)
        for c in source_range:
            mirrored_c = position - (c - position) if source_side == 'left' else position + (position - c)
            for r in range(rows):
                original_color = grid.values[r][c]
                if original_color == color1:
                    new_grid.values[r][mirrored_c] = color2
                elif original_color == color2:
                    new_grid.values[r][mirrored_c] = color1
                else:
                    new_grid.values[r][mirrored_c] = original_color
    
    return new_grid
