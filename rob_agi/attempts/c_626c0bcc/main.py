from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_626c0bcc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring sky-colored (8) regions with a specific pattern.
    
    The algorithm works as follows:
    1. Identify all connected sky-colored regions.
    2. Color each region using a specific pattern:
       - Place 2x2 blue (1) squares in corners where possible.
       - Fill remaining cells with a repeating pattern of red (2), green (3), yellow (4).
    3. Resolve any remaining color conflicts.
    
    This approach ensures no adjacent cells (including diagonally) have the same non-black color,
    while maintaining the overall shape of the original sky-colored regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sky_regions = find_connected_regions(input_grid, 8)
    
    for region in sky_regions:
        color_region(output_grid, region)
    
    resolve_all_conflicts(output_grid)
    return output_grid

def find_connected_regions(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and (r, c) not in visited:
                region = []
                queue = deque([(r, c)])
                while queue:
                    curr_r, curr_c = queue.popleft()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols:
                                queue.append((new_r, new_c))
                regions.append(region)
    return regions

def color_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    min_row, min_col = min(region)
    max_row, max_col = max(region)
    
    # Place 2x2 blue squares in corners where possible
    corners = [(min_row, min_col), (min_row, max_col-1), (max_row-1, min_col), (max_row-1, max_col-1)]
    for corner_r, corner_c in corners:
        if all((r, c) in region for r in range(corner_r, corner_r+2) for c in range(corner_c, corner_c+2)):
            for r in range(corner_r, corner_r+2):
                for c in range(corner_c, corner_c+2):
                    grid.set_cell(r, c, 1)  # Blue
    
    # Fill remaining cells with repeating pattern
    colors = [2, 3, 4]  # Red, Green, Yellow
    color_index = 0
    for r, c in region:
        if grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, colors[color_index])
            color_index = (color_index + 1) % 3

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
