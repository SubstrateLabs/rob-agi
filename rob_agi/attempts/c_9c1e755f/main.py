from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding patterns within rectangular regions.
    
    This function identifies seed patterns on the edges of the grid and uses them to define
    and fill rectangular regions. It expands patterns horizontally or vertically, respecting
    existing colored regions and grid boundaries. The process is repeated iteratively until
    no more expansions are possible. Finally, it fills any remaining black cells adjacent to
    colored cells.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns within regions.
    """
    grid = input_grid.deep_copy()
    
    while True:
        seed_patterns = identify_seed_patterns(grid)
        if not seed_patterns:
            break
        for pattern in seed_patterns:
            expand_and_fill(grid, pattern)
    
    fill_remaining(grid)
    return grid

def identify_seed_patterns(grid: ColoredGrid) -> List[Dict]:
    rows, cols = grid.get_dimensions()
    patterns = []
    
    # Check top and bottom edges
    for r in [0, rows-1]:
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                pattern = get_horizontal_pattern(grid, r, c)
                patterns.append({'edge': 'top' if r == 0 else 'bottom', 'pattern': pattern, 'row': r, 'col': c})
    
    # Check left and right edges
    for c in [0, cols-1]:
        for r in range(rows):
            if grid.get_cell(r, c) != 0:
                pattern = get_vertical_pattern(grid, r, c)
                patterns.append({'edge': 'left' if c == 0 else 'right', 'pattern': pattern, 'row': r, 'col': c})
    
    return sorted(patterns, key=lambda x: len(x['pattern']), reverse=True)

def get_horizontal_pattern(grid: ColoredGrid, r: int, start_c: int) -> List[int]:
    pattern = []
    cols = grid.get_dimensions()[1]
    for c in range(start_c, cols):
        color = grid.get_cell(r, c)
        if color == 0:
            break
        pattern.append(color)
    return pattern

def get_vertical_pattern(grid: ColoredGrid, start_r: int, c: int) -> List[int]:
    pattern = []
    rows = grid.get_dimensions()[0]
    for r in range(start_r, rows):
        color = grid.get_cell(r, c)
        if color == 0:
            break
        pattern.append(color)
    return pattern

def expand_and_fill(grid: ColoredGrid, seed: Dict):
    rows, cols = grid.get_dimensions()
    
    if seed['edge'] in ['top', 'bottom']:
        # Expand vertically
        start_row = 0 if seed['edge'] == 'top' else seed['row']
        end_row = seed['row'] if seed['edge'] == 'top' else rows
        for r in range(start_row, end_row):
            for c in range(seed['col'], min(cols, seed['col'] + len(seed['pattern']))):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, seed['pattern'][c - seed['col']])
    else:
        # Expand horizontally
        start_col = 0 if seed['edge'] == 'left' else seed['col']
        end_col = seed['col'] if seed['edge'] == 'left' else cols
        for c in range(start_col, end_col):
            for r in range(seed['row'], min(rows, seed['row'] + len(seed['pattern']))):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, seed['pattern'][r - seed['row']])

def fill_remaining(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = get_non_black_neighbors(grid, r, c)
                if neighbors:
                    grid.set_cell(r, c, neighbors[0])

def get_non_black_neighbors(grid: ColoredGrid, r: int, c: int) -> List[int]:
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
            neighbors.append(grid.get_cell(nr, nc))
    return neighbors
