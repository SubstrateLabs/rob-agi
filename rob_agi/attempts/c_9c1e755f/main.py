from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding patterns within rectangular regions.
    
    This function identifies seed patterns on the edges of the grid and uses them to define
    and fill rectangular regions. It expands patterns horizontally or vertically, respecting
    existing colored regions and grid boundaries. The process is done in a single pass,
    prioritizing longer patterns and filling remaining cells based on adjacent colors.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns within regions.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Identify and expand seed patterns
    seed_patterns = identify_seed_patterns(grid)
    for pattern in seed_patterns:
        expand_pattern(grid, pattern)
    
    # Fill remaining cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                fill_cell(grid, r, c)
    
    return grid

def identify_seed_patterns(grid: ColoredGrid) -> List[Dict]:
    rows, cols = grid.get_dimensions()
    patterns = []
    
    # Check all edges
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid.get_cell(r, c) != 0:
                if r == 0:
                    patterns.append({'edge': 'top', 'pattern': get_pattern(grid, r, c, 0, 1), 'row': r, 'col': c})
                elif r == rows-1:
                    patterns.append({'edge': 'bottom', 'pattern': get_pattern(grid, r, c, 0, 1), 'row': r, 'col': c})
                elif c == 0:
                    patterns.append({'edge': 'left', 'pattern': get_pattern(grid, r, c, 1, 0), 'row': r, 'col': c})
                elif c == cols-1:
                    patterns.append({'edge': 'right', 'pattern': get_pattern(grid, r, c, 1, 0), 'row': r, 'col': c})
    
    return sorted(patterns, key=lambda x: len(x['pattern']), reverse=True)

def get_pattern(grid: ColoredGrid, r: int, c: int, dr: int, dc: int) -> List[int]:
    pattern = []
    rows, cols = grid.get_dimensions()
    while 0 <= r < rows and 0 <= c < cols:
        color = grid.get_cell(r, c)
        if color == 0:
            break
        pattern.append(color)
        r += dr
        c += dc
    return pattern

def expand_pattern(grid: ColoredGrid, seed: Dict):
    rows, cols = grid.get_dimensions()
    pattern = seed['pattern']
    
    if seed['edge'] in ['top', 'bottom']:
        start_row = 0 if seed['edge'] == 'top' else seed['row']
        end_row = seed['row'] if seed['edge'] == 'top' else rows
        for r in range(start_row, end_row):
            for i, color in enumerate(pattern):
                c = seed['col'] + i
                if c < cols and grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, color)
    else:
        start_col = 0 if seed['edge'] == 'left' else seed['col']
        end_col = seed['col'] if seed['edge'] == 'left' else cols
        for c in range(start_col, end_col):
            for i, color in enumerate(pattern):
                r = seed['row'] + i
                if r < rows and grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, color)

def fill_cell(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
            grid.set_cell(r, c, grid.get_cell(nr, nc))
            break
