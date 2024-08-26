from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_626c0bcc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring sky-colored (8) regions with a specific pattern.
    
    The algorithm works as follows:
    1. Identify all connected sky-colored regions.
    2. Sort regions based on their top-left coordinate.
    3. For each region:
       a. Place a 2x2 blue (1) square in the top-left corner if possible.
       b. Place green (3) to the left of the blue area.
       c. Place yellow (4) to the right of the blue area.
       d. Fill remaining cells with red (2), ensuring no adjacent cells have the same color.
    4. Resolve any remaining color conflicts.
    
    This approach ensures no adjacent cells (including diagonally) have the same non-black color.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sky_regions = sorted(input_grid.find_connected_regions(8), key=lambda r: (min(r)[0], min(r)[1]))
    
    for region in sky_regions:
        color_region(output_grid, region)
    
    resolve_all_conflicts(output_grid)
    return output_grid

def color_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    start_row, start_col = min(region)
    end_row, end_col = max(region)
    width = end_col - start_col + 1
    height = end_row - start_row + 1
    
    # Place blue square
    place_shape(grid, region, (2, 2), 1, start_row, start_col)
    
    # Place green to the left
    if width > 2:
        place_shape(grid, region, (2, width - 2), 3, start_row, start_col + 2)
    
    # Place yellow below
    if height > 2:
        place_shape(grid, region, (height - 2, 2), 4, start_row + 2, start_col)
    
    # Fill remaining with red
    fill_gaps(grid, region, 2)

def place_shape(grid: ColoredGrid, region: List[Tuple[int, int]], shape: Tuple[int, int], color: int, row: int, col: int):
    for r in range(row, row + shape[0]):
        for c in range(col, col + shape[1]):
            if (r, c) in region:
                grid.set_cell(r, c, color)
                region.remove((r, c))

def find_next_uncolored(grid: ColoredGrid, region: List[Tuple[int, int]], row: int, col: int) -> Tuple[int, int]:
    for r, c in region:
        if r >= row and c >= col and grid.get_cell(r, c) == 0:
            return r, c
    return None

def fill_gaps(grid: ColoredGrid, region: List[Tuple[int, int]], color: int):
    for r, c in region:
        if grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, color)

def resolve_all_conflicts(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                resolve_conflict(grid, r, c)

def resolve_conflict(grid: ColoredGrid, row: int, col: int):
    current_color = grid.get_cell(row, col)
    neighbors = get_neighbors(grid, row, col)
    neighbor_colors = set(grid.get_cell(r, c) for r, c in neighbors if grid.get_cell(r, c) != 0)
    
    if current_color in neighbor_colors:
        for new_color in [1, 2, 3, 4]:
            if new_color not in neighbor_colors:
                grid.set_cell(row, col, new_color)
                break

def get_neighbors(grid: ColoredGrid, row: int, col: int) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbors.append((r, c))
    return neighbors
