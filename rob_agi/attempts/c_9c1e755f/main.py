from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding edge patterns.
    
    This function identifies non-black patterns on the edges of the grid and
    expands them to fill the largest possible rectangle. The expansion respects
    existing non-black cells and grid boundaries. Both vertical and horizontal
    expansions are handled. The process is repeated twice to capture any new
    patterns that emerge after the first expansion.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns.
    """
    grid = input_grid.deep_copy()
    
    # Perform the expansion process twice
    for _ in range(2):
        edge_patterns = find_edge_patterns(grid)
        for edge, start, pattern in edge_patterns:
            expand_pattern(grid, edge, start, pattern)
    
    return grid

def find_edge_patterns(grid: ColoredGrid) -> List[Tuple[str, Tuple[int, int], List[int]]]:
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
                cell = grid.get_cell(j, i) if is_vertical else grid.get_cell(i, j)
                if cell != 0:
                    if start is None:
                        start = (j, i) if is_vertical else (i, j)
                    pattern.append(cell)
                elif start is not None:
                    patterns.append((edge, start, pattern))
                    start = None
                    pattern = []
            if start is not None:
                patterns.append((edge, start, pattern))

    return patterns

def expand_pattern(grid: ColoredGrid, edge: str, start: Tuple[int, int], pattern: List[int]):
    rows, cols = grid.get_dimensions()
    row, col = start

    if edge in ["top", "bottom"]:
        # Expand vertically
        direction = 1 if edge == "top" else -1
        max_height = 0
        for r in range(row, rows if direction == 1 else -1, direction):
            if all(grid.get_cell(r, c) == 0 or grid.get_cell(r, c) == pattern[(c - col) % len(pattern)] for c in range(col, col + len(pattern))):
                max_height += 1
            else:
                break

        for r in range(row, row + max_height * direction, direction):
            for c in range(col, col + len(pattern)):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, pattern[(c - col) % len(pattern)])

    else:  # left or right
        # Expand horizontally
        direction = 1 if edge == "left" else -1
        max_width = 0
        for c in range(col, cols if direction == 1 else -1, direction):
            if all(grid.get_cell(r, c) == 0 or grid.get_cell(r, c) == pattern[(r - row) % len(pattern)] for r in range(row, row + len(pattern))):
                max_width += 1
            else:
                break

        for c in range(col, col + max_width * direction, direction):
            for r in range(row, row + len(pattern)):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, pattern[(r - row) % len(pattern)])
