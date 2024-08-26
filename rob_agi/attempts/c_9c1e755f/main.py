from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding patterns within rectangular regions.
    
    This function identifies non-black patterns on the edges of the grid and uses them to define
    rectangular regions. It then fills these regions by extending the edge patterns, prioritizing
    longer patterns and patterns from edges that define the region's boundaries. The process is
    repeated until no more expansions are possible, allowing for the creation and filling of new
    regions during the transformation.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns within regions.
    """
    grid = input_grid.deep_copy()
    while True:
        regions = identify_regions(grid)
        if not regions:
            break
        for region in regions:
            fill_region(grid, region)
    return grid

def identify_regions(grid: ColoredGrid) -> List[Dict]:
    rows, cols = grid.get_dimensions()
    regions = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                region = expand_region(grid, r, c, visited)
                if region:
                    regions.append(region)
    
    return regions

def expand_region(grid: ColoredGrid, start_r: int, start_c: int, visited: set) -> Dict:
    rows, cols = grid.get_dimensions()
    queue = deque([(start_r, start_c)])
    region = {'top': start_r, 'left': start_c, 'bottom': start_r, 'right': start_c}
    edge_patterns = {'top': [], 'bottom': [], 'left': [], 'right': []}

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        region['top'] = min(region['top'], r)
        region['left'] = min(region['left'], c)
        region['bottom'] = max(region['bottom'], r)
        region['right'] = max(region['right'], c)

        if r == region['top']:
            edge_patterns['top'].append(grid.get_cell(r, c))
        if r == region['bottom']:
            edge_patterns['bottom'].append(grid.get_cell(r, c))
        if c == region['left']:
            edge_patterns['left'].append(grid.get_cell(r, c))
        if c == region['right']:
            edge_patterns['right'].append(grid.get_cell(r, c))

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                queue.append((nr, nc))

    region['patterns'] = edge_patterns
    return region

def fill_region(grid: ColoredGrid, region: Dict):
    patterns = sorted(region['patterns'].items(), key=lambda x: len(x[1]), reverse=True)
    for edge, pattern in patterns:
        if not pattern:
            continue
        if edge in ['top', 'bottom']:
            for r in range(region['top'], region['bottom'] + 1):
                for c in range(region['left'], region['right'] + 1):
                    if grid.get_cell(r, c) == 0:
                        grid.set_cell(r, c, pattern[(c - region['left']) % len(pattern)])
        else:  # left or right
            for c in range(region['left'], region['right'] + 1):
                for r in range(region['top'], region['bottom'] + 1):
                    if grid.get_cell(r, c) == 0:
                        grid.set_cell(r, c, pattern[(r - region['top']) % len(pattern)])
