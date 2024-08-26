from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding edge patterns.
    
    This function identifies non-black patterns on the edges of the grid and
    expands them to fill the largest possible area. The expansion respects
    existing non-black cells and grid boundaries. The process works from the
    outside in, expanding patterns perpendicular to their orientation until
    they meet another pattern or the grid boundary. The process is repeated
    until no more expansions are possible.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns.
    """
    grid = input_grid.deep_copy()
    processed = set()
    
    while True:
        edge_patterns = find_edge_patterns(grid, processed)
        if not edge_patterns:
            break
        
        for edge, start, pattern in edge_patterns:
            expand_pattern(grid, edge, start, pattern, processed)
    
    return grid

def find_edge_patterns(grid: ColoredGrid, processed: Set[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int], List[int]]]:
    patterns = []
    rows, cols = grid.get_dimensions()

    # Check all edges
    for edge, range1, range2, is_vertical in [
        ("top", range(cols), [0], False),
        ("bottom", range(cols), [rows - 1], False),
        ("left", range(rows), [0], True),
        ("right", range(rows), [cols - 1], True)
    ]:
        for i in range1:
            start = None
            pattern = []
            for j in range2:
                r, c = (j, i) if is_vertical else (i, j)
                if (r, c) in processed:
                    continue
                cell = grid.get_cell(r, c)
                if cell != 0:
                    if start is None:
                        start = (r, c)
                    pattern.append(cell)
                elif start is not None:
                    patterns.append((edge, start, pattern))
                    start = None
                    pattern = []
            if start is not None:
                patterns.append((edge, start, pattern))

    return patterns

def expand_pattern(grid: ColoredGrid, edge: str, start: Tuple[int, int], pattern: List[int], processed: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    row, col = start
    queue = deque([(row, col)])

    while queue:
        r, c = queue.popleft()
        if (r, c) in processed:
            continue

        processed.add((r, c))
        if grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, pattern[(r - row) % len(pattern)] if edge in ["left", "right"] else pattern[(c - col) % len(pattern)])

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in processed:
                if edge in ["top", "bottom"] and nr != row:
                    queue.append((nr, nc))
                elif edge in ["left", "right"] and nc != col:
                    queue.append((nr, nc))
