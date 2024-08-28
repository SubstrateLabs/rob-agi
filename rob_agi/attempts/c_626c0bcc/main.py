from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_626c0bcc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring sky-colored (8) regions with a specific pattern.
    
    The algorithm works as follows:
    1. Identify all connected sky-colored regions.
    2. Color each region using a specific pattern:
       - Place a 2x2 blue (1) square in the top-left corner if possible.
       - Fill remaining cells in a clockwise spiral pattern with red (2), green (3), yellow (4).
    3. Apply symmetry if detected in the original pattern.
    4. Resolve any remaining color conflicts.
    
    This approach ensures no adjacent cells (including diagonally) have the same non-black color,
    while maintaining the overall shape and symmetry of the original sky-colored regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sky_regions = find_connected_regions(input_grid, 8)
    
    for region in sky_regions:
        color_region(output_grid, region)
    
    apply_symmetry(input_grid, output_grid)
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
    
    # Place 2x2 blue square in the top-left corner if possible
    if len(region) >= 4 and all((r, c) in region for r in range(min_row, min_row+2) for c in range(min_col, min_col+2)):
        for r in range(min_row, min_row+2):
            for c in range(min_col, min_col+2):
                grid.set_cell(r, c, 1)  # Blue
    
    # Fill remaining cells with specific pattern in clockwise spiral
    colors = [2, 3, 4]  # Red, Green, Yellow
    color_index = 0
    visited = set((r, c) for r in range(min_row, min_row+2) for c in range(min_col, min_col+2))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_index = 0
    r, c = min_row, min_col + 2 if len(region) >= 4 else min_col
    
    while len(visited) < len(region):
        if (r, c) in region and (r, c) not in visited:
            grid.set_cell(r, c, colors[color_index])
            visited.add((r, c))
            color_index = (color_index + 1) % 3
        
        nr, nc = r + directions[dir_index][0], c + directions[dir_index][1]
        if min_row <= nr <= max_row and min_col <= nc <= max_col and (nr, nc) in region and (nr, nc) not in visited:
            r, c = nr, nc
        else:
            dir_index = (dir_index + 1) % 4
            r, c = r + directions[dir_index][0], c + directions[dir_index][1]

def apply_symmetry(input_grid: ColoredGrid, output_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    
    # Check for vertical symmetry
    if all(input_grid.get_cell(r, c) == input_grid.get_cell(r, cols-1-c) for r in range(rows) for c in range(cols//2)):
        for r in range(rows):
            for c in range(cols//2):
                output_grid.set_cell(r, cols-1-c, output_grid.get_cell(r, c))
    
    # Check for horizontal symmetry
    elif all(input_grid.get_cell(r, c) == input_grid.get_cell(rows-1-r, c) for r in range(rows//2) for c in range(cols)):
        for r in range(rows//2):
            for c in range(cols):
                output_grid.set_cell(rows-1-r, c, output_grid.get_cell(r, c))

def resolve_all_conflicts(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) != 0:
                    if resolve_conflict(grid, r, c):
                        changed = True

def resolve_conflict(grid: ColoredGrid, row: int, col: int) -> bool:
    current_color = grid.get_cell(row, col)
    neighbors = get_neighbors(grid, row, col)
    neighbor_colors = set(grid.get_cell(r, c) for r, c in neighbors if grid.get_cell(r, c) != 0)
    
    if current_color in neighbor_colors:
        for new_color in [1, 2, 3, 4]:
            if new_color not in neighbor_colors:
                grid.set_cell(row, col, new_color)
                return True
    return False

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
