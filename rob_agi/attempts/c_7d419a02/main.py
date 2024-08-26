from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing some blue (8) regions to yellow (4).
    
    The transformation follows these rules:
    1. Large blue regions are framed with yellow, keeping the center blue.
    2. Small blue regions on the edges or adjacent to changed regions become yellow.
    3. Full blue columns on the edges of the grid become yellow.
    4. Black (0) and magenta (6) cells remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    regions = find_regions(grid)
    for region in regions:
        process_region(grid, region)
    handle_full_columns(grid)
    return grid

def is_blue(cell: int) -> bool:
    return cell == 8

def is_changeable(cell: int) -> bool:
    return cell == 8

def find_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    regions = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if is_blue(grid.values[r][c]) and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and is_blue(grid.values[curr_r][curr_c]):
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
                                stack.append((new_r, new_c))
                regions.append(region)
    return regions

def process_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    width = max_c - min_c + 1
    height = max_r - min_r + 1
    
    if width > 2 and height > 2:
        frame_width = min(width // 4, height // 4, 2)
        for r, c in region:
            if (r - min_r < frame_width or max_r - r < frame_width or
                c - min_c < frame_width or max_c - c < frame_width):
                grid.values[r][c] = 4  # Change to yellow
    else:
        if min_c == 0 or max_c == grid.num_cols - 1 or min_r == 0 or max_r == grid.num_rows - 1:
            for r, c in region:
                grid.values[r][c] = 4  # Change to yellow

def handle_full_columns(grid: ColoredGrid):
    for c in range(grid.num_cols):
        if all(is_blue(grid.values[r][c]) for r in range(grid.num_rows)):
            if c == 0 or c == grid.num_cols - 1:
                for r in range(grid.num_rows):
                    grid.values[r][c] = 4  # Change to yellow
