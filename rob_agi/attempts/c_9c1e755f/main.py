from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding patterns from edge seeds.
    
    This function identifies seed patterns on all edges of the grid and expands them
    perpendicular to their original orientation. It prioritizes longer patterns and
    respects existing colored regions and grid boundaries. The process involves:
    1. Identifying seed patterns on all edges
    2. Sorting patterns by length (longest first)
    3. Expanding patterns perpendicular to their edge, respecting existing non-zero cells
    4. Filling any remaining empty cells with adjacent colors

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns.
    """
    grid = input_grid.deep_copy()
    seed_patterns = identify_seed_patterns(grid)
    for pattern in seed_patterns:
        expand_pattern(grid, pattern)
    fill_remaining_cells(grid)
    return grid

def identify_seed_patterns(grid: ColoredGrid) -> List[Dict]:
    rows, cols = grid.get_dimensions()
    patterns = []
    
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid.get_cell(r, c) != 0:
                if r == 0:
                    patterns.append({'edge': 'top', 'pattern': get_pattern(grid, r, c, 0, 1), 'row': r, 'col': c})
                elif r == rows-1:
                    patterns.append({'edge': 'bottom', 'pattern': get_pattern(grid, r, c, 0, -1), 'row': r, 'col': c})
                elif c == 0:
                    patterns.append({'edge': 'left', 'pattern': get_pattern(grid, r, c, 1, 0), 'row': r, 'col': c})
                elif c == cols-1:
                    patterns.append({'edge': 'right', 'pattern': get_pattern(grid, r, c, -1, 0), 'row': r, 'col': c})
    
    return sorted(patterns, key=lambda x: len(x['pattern']), reverse=True)

def get_pattern(grid: ColoredGrid, r: int, c: int, dr: int, dc: int) -> List[int]:
    pattern = []
    rows, cols = grid.get_dimensions()
    while 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) != 0:
        pattern.append(grid.get_cell(r, c))
        r += dr
        c += dc
    return pattern

def expand_pattern(grid: ColoredGrid, seed: Dict):
    rows, cols = grid.get_dimensions()
    pattern = seed['pattern']
    
    if seed['edge'] in ['top', 'bottom']:
        start_row = 0 if seed['edge'] == 'top' else rows - 1
        end_row = rows if seed['edge'] == 'top' else -1
        step = 1 if seed['edge'] == 'top' else -1
        for r in range(start_row, end_row, step):
            if all(grid.get_cell(r, seed['col'] + i) == 0 for i in range(len(pattern))):
                for i, color in enumerate(pattern):
                    grid.set_cell(r, seed['col'] + i, color)
            else:
                break
    else:  # left or right
        start_col = 0 if seed['edge'] == 'left' else cols - 1
        end_col = cols if seed['edge'] == 'left' else -1
        step = 1 if seed['edge'] == 'left' else -1
        for c in range(start_col, end_col, step):
            if all(grid.get_cell(seed['row'] + i, c) == 0 for i in range(len(pattern))):
                for i, color in enumerate(pattern):
                    grid.set_cell(seed['row'] + i, c, color)
            else:
                break

def fill_remaining_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                        grid.set_cell(r, c, grid.get_cell(nr, nc))
                        break
