from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding patterns from edge seeds.
    
    This function identifies seed patterns on all edges of the grid and expands them
    perpendicular to their original orientation. It prioritizes patterns based on
    edge priority (left > top > right > bottom) and pattern length. The process involves:
    1. Identifying seed patterns on all edges
    2. Sorting patterns by edge priority and length
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
    
    # Left edge (highest priority)
    for r in range(rows):
        if grid.get_cell(r, 0) != 0:
            patterns.append({'edge': 'left', 'pattern': get_pattern(grid, r, 0, 0, 1), 'row': r, 'col': 0})
    
    # Top edge
    for c in range(cols):
        if grid.get_cell(0, c) != 0:
            patterns.append({'edge': 'top', 'pattern': get_pattern(grid, 0, c, 1, 0), 'row': 0, 'col': c})
    
    # Right edge
    for r in range(rows):
        if grid.get_cell(r, cols-1) != 0:
            patterns.append({'edge': 'right', 'pattern': get_pattern(grid, r, cols-1, 0, -1), 'row': r, 'col': cols-1})
    
    # Bottom edge (lowest priority)
    for c in range(cols):
        if grid.get_cell(rows-1, c) != 0:
            patterns.append({'edge': 'bottom', 'pattern': get_pattern(grid, rows-1, c, -1, 0), 'row': rows-1, 'col': c})
    
    # Sort patterns by edge priority and length
    edge_priority = {'left': 0, 'top': 1, 'right': 2, 'bottom': 3}
    return sorted(patterns, key=lambda x: (edge_priority[x['edge']], -len(x['pattern'])))

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
    
    if seed['edge'] in ['left', 'right']:
        start_col = seed['col']
        step = 1 if seed['edge'] == 'left' else -1
        for c in range(start_col, cols if step == 1 else -1, step):
            if all(grid.get_cell(r, c) == 0 for r in range(rows)):
                for r in range(rows):
                    grid.set_cell(r, c, pattern[r % len(pattern)])
            else:
                break
    else:  # top or bottom
        start_row = seed['row']
        step = 1 if seed['edge'] == 'top' else -1
        for r in range(start_row, rows if step == 1 else -1, step):
            if all(grid.get_cell(r, c) == 0 for c in range(cols)):
                for c in range(cols):
                    grid.set_cell(r, c, pattern[c % len(pattern)])
            else:
                break

def fill_remaining_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = []
                for dr, dc in [(0, -1), (-1, 0), (0, 1), (1, 0)]:  # Left, Up, Right, Down priority
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                        neighbors.append(grid.get_cell(nr, nc))
                        break  # Take the first non-zero neighbor based on priority
                if neighbors:
                    grid.set_cell(r, c, neighbors[0])
