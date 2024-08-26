from rob_agi.colored_grid import ColoredGrid
from collections import deque, Counter
from typing import List, Tuple, Dict

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Analyze the grid to identify non-black regions and extract pattern samples.
    2. Identify black regions and their surrounding contexts.
    3. Extend patterns into black regions based on surrounding contexts.
    4. Resolve conflicts and ensure consistency in pattern extensions.
    5. Handle edge cases and isolated black cells.
    6. Validate and refine the filled grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_neighbors(r: int, c: int, include_diagonal: bool = False) -> List[Tuple[int, int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc, grid.get_cell(nr, nc)))
        return neighbors

    def extract_pattern(r: int, c: int, size: int = 3) -> List[List[int]]:
        pattern = []
        for i in range(r, min(r + size, rows)):
            row = []
            for j in range(c, min(c + size, cols)):
                row.append(grid.get_cell(i, j))
            pattern.append(row)
        return pattern

    def find_best_pattern(r: int, c: int) -> List[List[int]]:
        patterns = []
        for nr, nc, color in get_neighbors(r, c, include_diagonal=True):
            if color != 0:
                patterns.append(extract_pattern(nr, nc))
        return max(patterns, key=lambda p: sum(row.count(0) for row in p), default=[])

    def extend_pattern(pattern: List[List[int]], r: int, c: int) -> None:
        pr, pc = len(pattern), len(pattern[0])
        for i in range(pr):
            for j in range(pc):
                if 0 <= r + i < rows and 0 <= c + j < cols and grid.get_cell(r + i, c + j) == 0:
                    grid.set_cell(r + i, c + j, pattern[i][j])

    # Step 1: Analyze grid and extract patterns
    patterns = {}
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                patterns[(r, c)] = extract_pattern(r, c)

    # Step 2 & 3: Identify black regions and extend patterns
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 0]
    for r, c in black_cells:
        best_pattern = find_best_pattern(r, c)
        if best_pattern:
            extend_pattern(best_pattern, r, c)

    # Step 4 & 5: Resolve conflicts and handle edge cases
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = get_neighbors(r, c)
                if neighbors:
                    color_counts = Counter(color for _, _, color in neighbors if color != 0)
                    if color_counts:
                        most_common_color = color_counts.most_common(1)[0][0]
                        grid.set_cell(r, c, most_common_color)

    # Step 6: Validate and refine
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                neighbors = get_neighbors(r, c, include_diagonal=True)
                if neighbors:
                    color_counts = Counter(color for _, _, color in neighbors if color != 0)
                    if color_counts:
                        most_common_color = color_counts.most_common(1)[0][0]
                        grid.set_cell(r, c, most_common_color)

    return grid
